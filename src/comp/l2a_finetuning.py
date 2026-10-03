"""SPEC-93: explicit physical bands, scaling and source masks."""
import numpy as np

EXPECTED_BANDS = {"blue": "B02", "green": "B03", "red": "B04", "nir08": "B8A",
                  "swir16": "B11", "swir22": "B12"}
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)


def asset_encoding(asset, key):
    bands = asset.get("eo:bands", [])
    raster = asset.get("raster:bands", [])
    if len(bands) != 1 or bands[0].get("name") != EXPECTED_BANDS[key]:
        raise ValueError(f"{key} is not the required physical band {EXPECTED_BANDS[key]}")
    if len(raster) != 1 or not all(k in raster[0] for k in ("scale", "offset", "nodata")):
        raise ValueError("asset needs explicit scale, offset and nodata")
    scale, offset, nodata = (float(raster[0][k]) for k in ("scale", "offset", "nodata"))
    if not np.isfinite([scale, offset, nodata]).all() or scale <= 0:
        raise ValueError("invalid reflectance encoding")
    return scale, offset, nodata


def physical_reflectance(raw, asset, key):
    scale, offset, nodata = asset_encoding(asset, key)
    raw = np.asarray(raw, np.float32)
    valid = np.isfinite(raw) & (raw != nodata)
    reflectance = np.where(valid, raw * scale + offset, -9999).astype(np.float32)
    return reflectance, valid


def l2a_valid(image, scl, labels):
    image, scl, labels = np.asarray(image), np.asarray(scl), np.asarray(labels)
    if image.shape != (6,) + scl.shape or labels.shape != scl.shape:
        raise ValueError("L2A/source labels are misaligned")
    return (np.isfinite(image).all(0) & (image != -9999).all(0)
            & ~np.isin(scl, SCL_INVALID) & (labels >= 0))


def l2a_water(image, scl):
    image, scl = np.asarray(image), np.asarray(scl)
    if image.shape != (6,) + scl.shape: raise ValueError("water mask grid differs")
    return (scl == 6) & ((image[1] - image[3]) / (image[1] + image[3] + 1e-6) > 0)
