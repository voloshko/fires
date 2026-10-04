"""SPEC-109: lossless binary caches and frozen development decision."""
import numpy as np

from .accuracy_contract import paired_group_comparison
from .russian_error_report import summarize


def pack_prediction(prediction):
    if prediction.dtype != bool or prediction.ndim != 3 or prediction.shape[-1] % 8:
        raise ValueError("boolean divisible-by-eight prediction grid required")
    return np.packbits(prediction, axis=-1, bitorder="little")


def unpack_prediction(packed, shape):
    if packed.dtype != np.uint8 or packed.shape != (shape[0], shape[1], shape[2]//8):
        raise ValueError("packed binary cache shape/type mismatch")
    return np.unpackbits(packed, axis=-1, bitorder="little").astype(bool)


def development_decision(datasets):
    if set(datasets) != {"russia_eval", "russia_cal", "western", "ems", "floga"}:
        raise ValueError("development population missing")
    reports = {}
    for name, rows in datasets.items():
        models = ["pair", "post", "reserve", "c1mm"]+(["prithvi"] if name in ("russia_eval", "russia_cal", "western") else [])
        reports[name] = dict(errors=summarize(rows, models), comparisons={m: paired_group_comparison(
            [r["group_id"] for r in rows], [r["pair"] for r in rows], [r[m] for r in rows]) for m in models[1:]})
    pooled = [r for name in ("russia_eval", "russia_cal", "western") for r in datasets[name]]
    pooled_errors = summarize(pooled, ["pair", "post", "reserve", "c1mm", "prithvi"])
    comparisons = {m: paired_group_comparison([r["group_id"] for r in pooled],
        [r["pair"] for r in pooled], [r[m] for r in pooled]) for m in ("post", "c1mm", "prithvi")}
    gain = all(c["delta"] >= .01 and c["ci95"][0] > 0 for c in comparisons.values())
    region = all(reports[n]["comparisons"][m]["delta"] >= -.01
        for n in ("russia_eval", "russia_cal", "western") for m in ("reserve", "c1mm"))
    mediterranean = all(reports[n]["comparisons"]["c1mm"]["delta"] >= -.01
        and reports[n]["comparisons"]["c1mm"]["ci95"][0] >= -.03 for n in ("ems", "floga"))
    misses = all(len(r["errors"]["models"]["pair"]["missed_positive_groups"])
        <= len(r["errors"]["models"]["c1mm"]["missed_positive_groups"]) for r in reports.values())
    return dict(eligible=bool(gain and region and mediterranean and misses),
        guards=dict(pooled_gain=gain, regional_points=region, mediterranean=mediterranean, missed_groups=misses),
        pooled=dict(comparisons=comparisons, errors=pooled_errors), datasets=reports)
