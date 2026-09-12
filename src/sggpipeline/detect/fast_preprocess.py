"""GPU preprocessing for OWLv2, replacing the CPU processor on the hot path.

The stock ``Owlv2ImageProcessor`` costs ~95ms per 1080p frame, ~4x the fp16
engine itself.  Profiling puts ~70% of that in a single ``torch.conv2d``: OWLv2
anti-aliases before downsampling (reproducing skimage's ``anti_aliasing=True``),
and the blur runs on the *padded* 1920x1920 float32 tensor on CPU.

This module runs the identical sequence on the GPU:

    uint8 -> float -> rescale -> pad to square -> gaussian blur -> resize -> normalize

The ordering is kept exactly as the reference implements it, including blurring
after padding, so the output matches rather than merely resembling it.
:func:`max_abs_difference` checks that claim instead of assuming it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class GpuOwlv2Preprocessor:
    """Preprocess frames on the GPU, returning the engine's input tensor.

    ``pad_value`` defaults to 0.0 to match current transformers.  Note that the
    original OWLv2 implementation pads with 0.5 grey; the two disagree, and the
    difference is a real accuracy variable rather than a formatting detail, so it
    is exposed here instead of hardcoded.
    """

    def __init__(
        self,
        image_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        pad_value: float = 0.0,
        mean: tuple[float, ...] = CLIP_MEAN,
        std: tuple[float, ...] = CLIP_STD,
        pad_last: bool = False,
    ):
        self.image_size = int(image_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.pad_value = float(pad_value)
        self.pad_last = bool(pad_last)
        self._mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self._kernel_cache: dict[tuple, torch.Tensor] = {}

    # -- public ---------------------------------------------------------------
    def __call__(self, images: list) -> torch.Tensor:
        """Return ``(B, 3, S, S)`` ready for the engine, on device."""
        # Each frame is moved to the GPU on its own and stacked there. Stacking
        # on the host first would copy every full-resolution frame an extra time
        # through CPU memory, which measurably outweighs the transfer itself.
        tensors = [
            self._to_tensor(image).unsqueeze(0).to(self.device, non_blocking=True)
            for image in images
        ]
        batch = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        return self.preprocess_tensor(batch)

    def preprocess_tensor(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the pipeline on a ``(B, 3, H, W)`` uint8 or float tensor."""
        x = batch.to(self.device, non_blocking=True).float()
        if batch.dtype == torch.uint8:
            x = x * (1.0 / 255.0)

        if self.pad_last:
            # Blur and resize the real pixels only, then pad at output
            # resolution. The downsampling factor is max(H,W)/S either way, so
            # the image content lands identically; only the blur's behaviour at
            # the padding seam differs. Costs ~1.8x fewer pixels on 16:9 input.
            x = self._resize_preserving_aspect(x)
            x = self._pad_to_square(x)
        else:
            x = self._pad_to_square(x)
            x = self._antialias(x)
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=False,
            )
        x = (x - self._mean) / self._std
        return x.to(self.dtype).contiguous()

    def _resize_preserving_aspect(self, x: torch.Tensor) -> torch.Tensor:
        """Scale so the long side becomes ``image_size``, keeping aspect ratio."""
        height, width = x.shape[-2:]
        side = max(height, width)
        scale = self.image_size / side
        target = (max(1, round(height * scale)), max(1, round(width * scale)))
        x = self._antialias(x, target)
        return F.interpolate(
            x, size=target, mode="bilinear", align_corners=False, antialias=False
        )

    # -- stages ---------------------------------------------------------------
    def _to_tensor(self, image) -> torch.Tensor:
        """PIL or HWC array -> CHW uint8 tensor, without a CPU float conversion."""
        array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)

    def _pad_to_square(self, x: torch.Tensor) -> torch.Tensor:
        """Pad bottom/right to a square, which is what makes box rescaling a single factor."""
        height, width = x.shape[-2:]
        side = max(height, width)
        if height == side and width == side:
            return x
        return F.pad(x, (0, side - width, 0, side - height), value=self.pad_value)

    def _antialias(self, x: torch.Tensor, target: tuple[int, int] | None = None) -> torch.Tensor:
        """Gaussian pre-filter matched to the downsampling factor.

        Skipping this is the tempting optimization and the wrong one: without it
        a 2x downsample aliases high-frequency texture, which changes detector
        scores on exactly the small objects AG cares about.
        """
        height, width = x.shape[-2:]
        target_h, target_w = target or (self.image_size, self.image_size)
        factors = (height / target_h, width / target_w)
        sigma = tuple(max((f - 1.0) / 2.0, 0.0) for f in factors)
        if sigma[0] <= 0 and sigma[1] <= 0:
            return x

        kernel_y, kernel_x = self._gaussian_kernels(sigma)
        pad_y, pad_x = kernel_y.numel() // 2, kernel_x.numel() // 2
        channels = x.shape[1]

        # Separable: two 1-D passes instead of one 2-D convolution.
        x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
        x = F.conv2d(x, kernel_y.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
        x = F.conv2d(x, kernel_x.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
        return x

    def _gaussian_kernels(self, sigma: tuple[float, float]):
        key = (round(sigma[0], 6), round(sigma[1], 6))
        if key not in self._kernel_cache:
            self._kernel_cache[key] = tuple(
                self._gaussian_1d(s) for s in sigma
            )
        return self._kernel_cache[key]

    def _gaussian_1d(self, sigma: float) -> torch.Tensor:
        """1-D Gaussian sized as the reference does: ``2*ceil(3*sigma)+1``."""
        if sigma <= 0:
            return torch.ones(1, device=self.device)
        radius = int(np.ceil(3 * sigma))
        positions = torch.arange(
            -radius, radius + 1, device=self.device, dtype=torch.float32
        )
        kernel = torch.exp(-(positions**2) / (2 * sigma**2))
        return kernel / kernel.sum()


def max_abs_difference(fast: torch.Tensor, reference: np.ndarray) -> dict:
    """Agreement between the GPU pipeline and the stock processor's output."""
    a = fast.detach().float().cpu().numpy()
    b = np.asarray(reference, dtype=np.float32)
    if a.shape != b.shape:
        return {"error": f"shape mismatch {a.shape} vs {b.shape}"}
    diff = np.abs(a - b)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "fraction_above_1e-3": float((diff > 1e-3).mean()),
    }
