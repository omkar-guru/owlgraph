"""`sgg prepare` for int8 must stream calibration frames, not preload them.

Preloading 384 frames at 960px fp32 held ~4 GB in host memory and crashed a 15 GB
machine. Everything heavy is mocked; no GPU, model or dataset is needed.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from sggpipeline import cli
from sggpipeline.detect.quantize import CalibrationReader


def fake_frames(n_videos=4, per_video=3):
    return [SimpleNamespace(video_id=f"v{v}", frame_key=f"v{v}.mp4/{i:06d}.png",
                            image_path=Path(f"/frames/v{v}/{i}.png"))
            for v in range(n_videos) for i in range(per_video)]


class FakeAG:
    def __init__(self, *args, **kwargs):
        self.frames = fake_frames()

    def load(self):
        return self

    def __len__(self):
        return len(self.frames)


class StreamingCalibrationTest(unittest.TestCase):
    def test_prepare_int8_streams_and_records_manifest(self):
        loaded, captured, manifest = [], {}, {}

        def load_image(path):
            loaded.append(path)
            return object()

        def to_int8_onnx(src, dst, reader, **kwargs):
            captured["reader"] = reader
            captured["loaded_before_iterating"] = len(loaded)
            batches = []
            while (batch := reader.get_next()) is not None:
                batches.append(batch["pixel_values"])
            captured["batches"] = batches
            return dst

        def write_report(payload, path):
            manifest.update(payload)
            return path

        args = SimpleNamespace(ag_root="/ag", calib_split="train", eval_split="test",
                               calib_frames=5, device="cpu", calib_method="entropy")
        variant = SimpleNamespace(checkpoint="base", name="large_int8")
        with tempfile.TemporaryDirectory() as tmp:
            ws = SimpleNamespace(result=lambda name: Path(tmp) / name)
            with mock.patch("sggpipeline.ag.dataset.ActionGenome", FakeAG), \
                 mock.patch("sggpipeline.detect.owlv2.load_owlv2", return_value=(None, None)), \
                 mock.patch("sggpipeline.pipeline.build_preprocessor",
                            return_value=lambda images: np.zeros((1, 3, 8, 8))), \
                 mock.patch("sggpipeline.detect.preprocess.load_image", load_image), \
                 mock.patch("sggpipeline.pipeline.load_queries",
                            return_value=SimpleNamespace(embeds=np.zeros((2, 4), np.float32))), \
                 mock.patch("sggpipeline.detect.quantize.to_int8_onnx", to_int8_onnx), \
                 mock.patch("sggpipeline.pipeline.write_report", write_report):
                cli._prepare_int8(args, variant, ws, Path("src.onnx"), Path("dst.onnx"))

        reader = captured["reader"]
        self.assertIsInstance(reader, CalibrationReader)
        self.assertIsNone(reader.pixel_values)
        self.assertEqual(len(reader.image_paths), 5)
        self.assertEqual(captured["loaded_before_iterating"], 0)
        self.assertEqual(len(captured["batches"]), 5)
        self.assertTrue(all(b.shape == (1, 3, 8, 8) and b.dtype == np.float32
                            for b in captured["batches"]))
        self.assertEqual(len(loaded), 5)
        # Sampled across videos, and exactly those frames are in the manifest.
        self.assertEqual(len(manifest["frame_keys"]), 5)
        self.assertEqual({k.split("/")[0] for k in manifest["frame_keys"]},
                         {f"v{v}.mp4" for v in range(4)})
        self.assertEqual([Path(f"/frames/{k.split('.mp4/')[0]}/{int(k.split('/')[1][:6])}.png")
                          for k in manifest["frame_keys"]], reader.image_paths)

    def test_refuses_to_calibrate_on_evaluation_split(self):
        args = SimpleNamespace(ag_root="/ag", calib_split="test", eval_split="test",
                               calib_frames=5, device="cpu", calib_method="entropy")
        with self.assertRaises(SystemExit):
            cli._prepare_int8(args, SimpleNamespace(checkpoint="base", name="x"), None,
                              Path("a"), Path("b"))


if __name__ == "__main__":
    unittest.main()
