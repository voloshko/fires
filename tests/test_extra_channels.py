"""SPEC-48/49: хуки дополнительных каналов включаются только переменными окружения."""
import os
import numpy as np


def _chip():
    from src.comp.chips import BsChip
    pre = np.random.default_rng(0).integers(500, 3000, (10, 40, 40)).astype(np.float32); post = pre.copy(); post[8, :10] += 1500
    return BsChip("x", pre, post, np.zeros((3, 40, 40), np.float32), np.zeros((2, 40, 40), np.float32), None)


def test_без_переменных_вход_прежний(monkeypatch):
    monkeypatch.delenv("EXTRA_CHANNELS_DIR", raising=False); monkeypatch.delenv("LOCAL_Z", raising=False)
    from src.comp.hypothesis_lab import bs_inputs, OPTICAL
    assert bs_inputs(_chip(), "optical").shape[0] == len(OPTICAL)


def test_local_z_и_каталог_карт(monkeypatch, tmp_path):
    from src.comp.hypothesis_lab import bs_inputs, OPTICAL
    np.save(tmp_path / "x.npy", np.full((40, 40, 4), 0.25, np.float16))
    monkeypatch.setenv("EXTRA_CHANNELS_DIR", str(tmp_path)); monkeypatch.setenv("LOCAL_Z", "1")
    x = bs_inputs(_chip(), "optical")
    assert x.shape[0] == len(OPTICAL) + 4 + 2 and np.isfinite(x).all()
    assert abs(x[len(OPTICAL)].mean() - 0.25) < 1e-3
