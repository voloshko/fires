"""Веса сетей в ансамбле: длина проверяется, среднее взвешенное считается."""
import numpy as np
import pytest


def test_weighted_average_matches_numpy():
    probs = [np.full((2, 2, 4), 0.2), np.full((2, 2, 4), 0.8)]
    assert np.allclose(np.average(probs, axis=0, weights=[3, 1]), 0.35)


def test_length_mismatch_rejected():
    from src.comp import ensemble

    with pytest.raises(ValueError, match="весов"):
        ensemble.predict([object(), object()], None, None, net_weights=[1.0])
