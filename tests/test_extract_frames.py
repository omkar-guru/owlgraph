"""Frame extraction must never trust a half-written file on resume."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sggpipeline.ag.extract_frames import _complete


class CompletenessTest(unittest.TestCase):
    def test_truncated_png_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.png"
            Image.new("RGB", (32, 24), (10, 20, 30)).save(good)
            self.assertTrue(_complete(good, "png"))
            bad = Path(tmp) / "bad.png"
            bad.write_bytes(good.read_bytes()[:-20])
            self.assertFalse(_complete(bad, "png"))
            empty = Path(tmp) / "empty.png"
            empty.write_bytes(b"")
            self.assertFalse(_complete(empty, "png"))
            self.assertFalse(_complete(Path(tmp) / "missing.png", "png"))

    def test_other_formats_are_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.jpg"
            f.write_bytes(b"x")
            self.assertTrue(_complete(f, "jpg"))


if __name__ == "__main__":
    unittest.main()
