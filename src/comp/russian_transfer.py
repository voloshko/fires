"""SPEC-92: split-safe background mining from published interval perimeters."""
import math
import numpy as np
from src.comp.accuracy_contract import canonical_hash


def hard_background_weights(records, false_positive_counts):
    if any(r["partition"] != "fit" for r in records):
        raise ValueError("hard-negative mining may only see fit records")
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(false_positive_counts):
        raise ValueError("mining counts do not match the fit population")
    if any(not np.isfinite(v) or v < 0 or int(v) != v for v in false_positive_counts.values()):
        raise ValueError("invalid false-positive counts")
    ranked = sorted(ids, key=lambda sid: (-false_positive_counts[sid], sid))
    hard = set(ranked[:math.ceil(len(ranked) * .25)])
    return {sid: 2 if sid in hard else 1 for sid in ids}


def transfer_acceptance(russian_comparison, stability, competitors):
    accepted = russian_comparison["delta"] >= .01 and russian_comparison["ci95"][0] > 0
    accepted &= all(c["delta"] >= -.01 for c in stability.values())
    if set(competitors) != {"c1mm", "prithvi"}: raise ValueError("all named competitors are required")
    wins = {name: c["delta"] > 0 and c["ci95"][0] > 0 for name, c in competitors.items()}
    return dict(accepted=bool(accepted), competitor_superiority=all(wins.values()),
                per_competitor_superiority=wins)


def validate_pair_initializer(bundle, seed, base_manifest, expected_recipe):
    if (bundle.get("variant") != "pair" or bundle.get("seed") != seed or bundle.get("epochs") != 120
            or bundle.get("smoke") or bundle.get("width") != 32 or bundle.get("depth") != 7
            or bundle.get("batch") != 8 or bundle.get("data_manifest") != base_manifest):
        raise ValueError("initializer does not match frozen SPEC-91 base")
    fields = ("source_hashes", "software", "data_manifest", "fit_ids", "width", "depth", "epochs", "batch", "evaluation_ids")
    if any(k not in bundle for k in fields): raise ValueError("initializer lacks archived recipe fields")
    recipe = canonical_hash({k: bundle[k] for k in fields})
    if recipe != expected_recipe: raise ValueError("initializer source recipe does not match SPEC-91")
    return recipe
