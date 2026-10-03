import numpy as np

from scripts.forest_pair_train import normalization
from scripts.prepare_forest_pairs import hls_bad


def test_fit_normalization_ignores_invalid_values_and_shares_dates():
    valid = np.ones((8, 8), bool); valid[:4] = False
    x = np.ones((12, 8, 8), np.float32); x[6:] = 3; x[:, :4] = -9999
    samples = [(x, np.zeros((8, 8), np.int64), valid)]
    mean, std = normalization(samples, "pair")
    np.testing.assert_allclose(mean, np.full(12, 2))
    np.testing.assert_allclose(std, np.ones(12))
    mean, _ = normalization([(x[6:], samples[0][1], valid)], "post")
    np.testing.assert_allclose(mean, np.full(6, 3))


def test_hls_qc_masks_snow_shadow_cloud_and_preserves_water():
    fmask = np.array([0, 1, 2, 4, 8, 16, 32, 255], np.uint8)
    np.testing.assert_array_equal(hls_bad(fmask), [False, True, True, True, True, True, False, True])
