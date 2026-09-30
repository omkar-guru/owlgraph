"""Export the OWLv2 detection graph to ONNX as the front half of the TRT path.

Export always runs in fp32 on CPU-visible weights: the engine's precision is a
TensorRT build-time decision, so baking fp16 into the ONNX file would only lose
range before the builder ever sees the network.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .owlv2 import Owlv2DetectionGraph

INPUT_NAMES = ["pixel_values", "query_embeds"]
OUTPUT_NAMES = ["pred_logits", "pred_boxes", "objectness"]


def export_onnx(
    model,
    num_prompts: int,
    image_size: int,
    out_path: Path,
    opset: int = 17,
    device: str = "cuda",
    batch_size: int = 1,
    native_image_size: int | None = None,
    with_features: bool = False,
) -> Path:
    """Trace the detection graph to ONNX with fully static shapes.

    Static shapes are intentional: batch is 1 for per-frame streaming, the prompt
    count is fixed by the AG vocabulary, and a single optimization profile lets
    TensorRT specialize the attention kernels instead of hedging across shapes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Off-native input sizes need the learned position grid resampled. This is
    # done once, into the weights, so the exported graph stays static.
    if native_image_size is not None and image_size != native_image_size:
        from .owlv2 import retarget_resolution

        retarget_resolution(model, image_size)
    graph = Owlv2DetectionGraph(model, interpolate_pos_encoding=False,
                                with_features=with_features).eval().to(device)
    embed_dim = model.config.text_config.hidden_size

    dummy_pixels = torch.randn(batch_size, 3, image_size, image_size, device=device)
    dummy_query = torch.randn(num_prompts, embed_dim, device=device)
    dummy_query = dummy_query / dummy_query.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        torch.onnx.export(
            graph,
            (dummy_pixels, dummy_query),
            str(out_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES + (["patch_features"] if with_features else []),
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    return out_path


@torch.no_grad()
def check_parity(model, processor, graph_outputs, pixel_values, query_embeds) -> dict:
    """Compare the decomposed graph against the stock HF forward pass.

    The graph re-implements what ``Owlv2ForObjectDetection.forward`` does
    internally.  If that re-implementation drifts, mAP drops for a reason that
    looks like quantization damage but is not, so verify it explicitly.
    """
    logits, boxes, objectness = graph_outputs
    reference = Owlv2DetectionGraph(model)(pixel_values, query_embeds)
    ref_logits, ref_boxes, ref_obj = reference

    def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a.float() - b.float()).abs().max())

    return {
        "logits_max_abs_diff": max_abs(logits, ref_logits),
        "boxes_max_abs_diff": max_abs(boxes, ref_boxes),
        "objectness_max_abs_diff": max_abs(objectness, ref_obj),
    }
