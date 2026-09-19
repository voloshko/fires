"""Признаки BS-чипа (SPEC-18/30)."""
import numpy as np

from src.comp.chips import BsChip
from src.comp.features import normalise_post


def test_normalise_post_возвращает_pre_вне_гари():
    """post = 1.2·pre + 100 на всём чипе → после нормализации post ≈ pre (SPEC-30)."""
    rng = np.random.default_rng(0)
    pre = rng.integers(500, 3000, (10, 64, 64)).astype(np.uint16); pre[9] = 4
    post = (pre.astype(np.float32) * 1.2 + 100).astype(np.uint16); post[9] = 4
    chip = BsChip("x", pre, post, np.zeros((3, 64, 64)), np.zeros((2, 64, 64)), None)
    out = normalise_post(chip)
    rel = np.abs(out.post[:9].astype(float) - pre[:9]) / pre[:9]
    assert rel.mean() < 0.01 and (out.post[9] == 4).all()
