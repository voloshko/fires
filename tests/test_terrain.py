import numpy as np
from src.comp.terrain import water, cos_incidence, c_correct, slope_deg

def test_water_bit():
    f = np.array([0, 32, 34, 255, 2], np.uint8); assert water(f).tolist() == [False, True, True, False, False]

def test_flat_incidence_equals_cos_sza():
    dem = np.zeros((20, 20)); ci = cos_incidence(dem, np.full((20, 20), 40.0), np.full((20, 20), 150.0))
    assert np.allclose(ci, np.cos(np.radians(40)))

def test_slope_facing_sun_is_brighter():
    x = np.arange(40) * 30.0; dem = np.tile(-x * 0.5, (40, 1))   # высота падает к востоку: склон смотрит на восток
    east = cos_incidence(dem, np.full(dem.shape, 45.0), np.full(dem.shape, 90.0)).mean()
    west = cos_incidence(dem, np.full(dem.shape, 45.0), np.full(dem.shape, 270.0)).mean()
    assert east > np.cos(np.radians(45)) > west

def test_c_correction_flattens_shading():
    rng = np.random.default_rng(0); ci = rng.uniform(0.2, 1.0, (100, 100)); X = (0.05 + 0.2 * ci)[None].repeat(2, 0)
    ok = np.ones_like(ci, bool); sl = np.full_like(ci, 20.0)
    Y, used = c_correct(X, ci, 40.0, ok, sl)
    assert used == [0, 1] and Y[0].std() < 0.1 * X[0].std()

def test_water_ndwi_keeps_dark_burn():
    from src.comp.terrain import water_ndwi
    f = np.array([32, 32, 0]); g = np.array([0.05, 0.03, 0.05]); n = np.array([0.02, 0.06, 0.02])   # вода, тёмная гарь под «водой», суша
    assert water_ndwi(f, g, n).tolist() == [True, False, False]
