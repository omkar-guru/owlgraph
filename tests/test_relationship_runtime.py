"""The streaming relationship predictor: padded/graph paths equal the plain computation."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sggpipeline.ag.classes import AG_OBJECT_CLASSES
from sggpipeline.ag.relations import PREDICATES
from sggpipeline.relations.model import RelationshipHead, all_pair_geometry
from sggpipeline.relations.runtime import RelationshipPredictor


@torch.no_grad()
def reference(model, packed, feats, size, budget):
    """Unpadded head on exactly n detections, as a plain reading of the method."""
    n = len(packed)
    subj, obj = model.roles(feats[None].float(), packed[None, :, 5].long())
    geo = all_pair_geometry(packed[None, :, :4], torch.tensor([size], device=packed.device))
    route = model.route(subj, obj, geo)[0].masked_fill(torch.eye(n, dtype=torch.bool, device=packed.device),
                                                       float("-inf"))
    top = torch.topk(route.flatten(), min(budget, n * (n - 1)))
    si, oi = top.indices // n, top.indices % n
    probs = model.probabilities(model.classify(subj[0, si], obj[0, oi], geo[0, si, oi]))
    return si.cpu().numpy(), oi.cpu().numpy(), torch.sigmoid(top.values).cpu().numpy(), probs.cpu().numpy()


def detections(n, rng, device):
    xy = torch.from_numpy(rng.uniform(0, 400, (n, 2))).float()
    boxes = torch.cat([xy, xy + torch.from_numpy(rng.uniform(10, 200, (n, 2))).float()], 1)
    packed = torch.zeros(n, 8)
    packed[:, :4] = boxes
    packed[:, 4] = torch.from_numpy(np.sort(rng.uniform(0.05, 1, n))[::-1].copy()).float()
    packed[:, 5] = torch.from_numpy(rng.integers(0, len(AG_OBJECT_CLASSES), n)).float()
    feats = torch.from_numpy(rng.normal(0, 1, (n, 768))).half()
    return packed.to(device), feats.to(device)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class RuntimeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        embeds = torch.nn.functional.normalize(torch.randn(len(PREDICATES), 512), dim=1)
        self.tmp = tempfile.TemporaryDirectory()
        self.ckpt = Path(self.tmp.name) / "head.pt"
        torch.save(RelationshipHead(768, len(AG_OBJECT_CLASSES), embeds).state_dict(), self.ckpt)
        self.embeds = embeds.numpy()

    def tearDown(self):
        self.tmp.cleanup()

    def test_graph_and_eager_match_reference(self):
        self.check()

    def test_text_label_head(self):
        class_embeds = torch.nn.functional.normalize(torch.randn(len(AG_OBJECT_CLASSES), 512), dim=1)
        torch.save(RelationshipHead(768, len(AG_OBJECT_CLASSES), torch.from_numpy(self.embeds),
                                    class_embeds=class_embeds).state_dict(), self.ckpt)
        self.check()

    def check(self):
        rng = np.random.default_rng(0)
        graph = RelationshipPredictor(self.ckpt, self.embeds, pair_budget=64, cuda_graph=True)
        eager = RelationshipPredictor(self.ckpt, self.embeds, pair_budget=64, cuda_graph=False)
        # Varying detection counts exercise padding, including budgets padding would overfill.
        for n in (32, 5, 20, 2, 32, 9):
            packed, feats = detections(n, rng, "cuda")
            result = SimpleNamespace(device_packed=packed, device_features=feats, image_size=(640, 480))
            ref_s, ref_o, ref_p, ref_probs = reference(graph.model, packed, feats, (640, 480), 64)
            for predictor in (graph, eager):
                out = predictor(result)
                self.assertEqual(len(out.subject), len(ref_s))
                np.testing.assert_array_equal(out.subject, ref_s)
                np.testing.assert_array_equal(out.object, ref_o)
                np.testing.assert_allclose(out.pair_score, ref_p, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(out.predicate_probs, ref_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
