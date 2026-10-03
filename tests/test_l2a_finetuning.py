import numpy as np
import pytest
from src.comp.l2a_finetuning import asset_encoding, l2a_valid, l2a_water, physical_reflectance


def asset(band="B8A", offset=-.1):
    return {"eo:bands": [{"name": band}], "raster:bands": [{"scale": .0001, "offset": offset, "nodata": 0}]}


def test_asset_metadata_controls_scaling_and_nodata():
    image, valid = physical_reflectance([0, 2000, 3000], asset(), "nir08")
    np.testing.assert_allclose(image, [-9999, .1, .2], atol=1e-6)
    np.testing.assert_array_equal(valid, [False, True, True])
    image, _ = physical_reflectance([2000], asset(offset=0), "nir08")
    np.testing.assert_allclose(image, [.2])


def test_broad_nir_and_unknown_scale_are_rejected():
    with pytest.raises(ValueError): asset_encoding(asset("B08"), "nir08")
    with pytest.raises(ValueError): asset_encoding({"eo:bands": [{"name": "B8A"}]}, "nir08")


def test_snow_cloud_and_unknown_pixels_are_not_training_targets():
    scl = np.array([[4, 6, 8, 11, 0, 5]], np.uint8)
    image = np.ones((6, 1, 6), np.float32); label = np.array([[1, 0, 1, 1, 0, -1]])
    np.testing.assert_array_equal(l2a_valid(image, scl, label), [[True, True, False, False, False, False]])


def test_water_requires_both_scl_and_ndwi():
    image = np.ones((6, 1, 3), np.float32); image[1] = [2, 2, .5]
    np.testing.assert_array_equal(l2a_water(image, [[6, 4, 6]]), [[True, False, False]])
