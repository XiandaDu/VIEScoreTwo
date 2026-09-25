#!/usr/bin/env python
"""Repair preprocessing tests (R0-1 aspect ratio, R0-3 token truncation).

Run: python tests/test_repair_geometry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viescore2"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from repair_quant import (  # noqa: E402
    fit_for_inpaint, unfit_from_inpaint, clip_truncate, grid_to_mask,
    INPAINT_MULT,
)


def test_roundtrip_sizes():
    for W, H in [(640, 480), (333, 777), (1024, 1024), (2000, 500), (100, 100)]:
        img = Image.new("RGB", (W, H), (10, 20, 30))
        grid = np.zeros((16, 16), dtype=np.uint8)
        grid[3:6, 4:9] = 1
        fit_img, fit_mask, wh = fit_for_inpaint(img, grid)
        w, h = wh
        # aspect ratio preserved on the content region (±1px rounding)
        assert abs(w / h - W / H) < 0.01, (W, H, wh)
        # canvas is backend-friendly
        assert fit_img.size[0] % INPAINT_MULT == 0
        assert fit_img.size[1] % INPAINT_MULT == 0
        assert fit_img.size == fit_mask.size
        # padding is never masked
        m = np.array(fit_mask)
        assert m[h:, :].sum() == 0 and m[:, w:].sum() == 0, "mask leaked into padding"
        # round trip restores original size
        back = unfit_from_inpaint(fit_img, wh, (W, H))
        assert back.size == (W, H)
    print("ok: roundtrip sizes + padding hygiene")


def test_mask_alignment():
    # The padded mask, cropped+resized back, must land on the same cells as a
    # mask drawn directly at native resolution (IoU ≈ 1 up to resampling).
    W, H = 613, 411
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[2, 3] = 1
    grid[10:13, 8:12] = 1
    img = Image.new("RGB", (W, H))
    _, fit_mask, wh = fit_for_inpaint(img, grid)
    back = np.array(unfit_from_inpaint(fit_mask.convert("RGB"), wh, (W, H)).convert("L")) > 127
    native = np.array(grid_to_mask(grid, W, H)) > 127
    inter = (back & native).sum()
    union = (back | native).sum()
    assert inter / union > 0.95, f"mask misaligned after roundtrip: IoU={inter/union:.3f}"
    print("ok: mask alignment")


class FakeTok:
    """Whitespace tokenizer with CLIP-style bos/eos ids for clip_truncate."""
    class _Enc:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text):
        words = text.split()
        return self._Enc([0] + list(range(1, len(words) + 1)) + [99])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


def test_clip_truncate():
    tok = FakeTok()
    short, trunc = clip_truncate(tok, "a b c", max_tokens=75)
    assert short == "a b c" and trunc is False
    long_text = " ".join(f"tok{i}" for i in range(200))
    out, trunc = clip_truncate(tok, long_text, max_tokens=75)
    assert trunc is True
    assert len(out.split()) == 75, f"expected 75 tokens kept, got {len(out.split())}"
    print("ok: clip truncation at token boundary")


if __name__ == "__main__":
    test_roundtrip_sizes()
    test_mask_alignment()
    test_clip_truncate()
    print("ALL PASS: test_repair_geometry")
