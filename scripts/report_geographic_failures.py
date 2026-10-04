"""SPEC-106: CPU-only attribution; refuse missing or changed prediction caches."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import experiment_fingerprint, sha256_file, validate_manifest, verify_cache
from src.comp.accuracy_artifacts import execution_sources
from src.comp.geographic_failure_report import (
    diagnostic_masks, group_influence, probability_max, spectral_indices,
    spectral_summary, verify_recipe, verify_rows,
)
from src.comp.l2a_finetuning import l2a_water
from scripts.forest_pair_train import read_sample
from scripts import score_russian_transfer
from scripts.score_russian_transfer import counts_for

PARENT_SHA = "749699f3f6e01f0a0a6576afb496c891c4bc33c05ad1cca4087585ed5628cb2c"


def load_completed_prediction(root, records, paths, kind, cache, prithvi_root):
    """Match the original signature without any neural forward path."""
    if not cache.exists() or not cache.with_name(cache.name+".binding.json").exists():
        raise ValueError("missing completed cache; inference forbidden")
    files = [root/r["files"][k] for r in records for k in sorted(r["files"])]
    sources = execution_sources(Path(score_russian_transfer.__file__),
        "scripts/russian_transfer_train.py", "src/comp/russian_transfer.py", "scripts/forest_pair_train.py")
    if kind == "prithvi":
        paths = [prithvi_root/"Prithvi_EO_V2_300M_BurnScars.pt", prithvi_root/"burn_scars_config.yaml"]
    signature = experiment_fingerprint(records, files, paths, sources,
        dict(kind=kind, tta=8, precision="bf16"))
    verify_cache(cache, signature)
    array = np.load(cache, mmap_mode="r", allow_pickle=False)
    if array.shape != (len(records), records[0]["grid"]["height"], records[0]["grid"]["width"]):
        raise ValueError("completed cache geometry differs")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "manifest", "parent", "runs", "reference-root", "prithvi-root", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable diagnosis already exists")
    if sha256_file(args.parent) != PARENT_SHA:
        raise ValueError("wrong completed geographic parent")
    parent = json.loads(args.parent.read_text())
    if parent["spec_id"] != "SPEC-105" or sha256_file(args.manifest) != parent["manifest_sha256"]:
        raise ValueError("diagnostic source population differs")
    recipe_file = args.parent.with_suffix(".recipe.json")
    recipe = json.loads(recipe_file.read_text())
    verify_recipe(recipe, parent["recipe_id"])
    for name, digest in recipe["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("parent numeric source changed")
    payload = json.loads(args.manifest.read_text())
    records = payload["records"]
    validate_manifest(records)
    for record in records:
        for name, relative in record["files"].items():
            if sha256_file(args.data_root/relative) != record["hashes"][name]:
                raise ValueError("parent pixels changed")
    models = dict(reserve=[args.runs/f"reserve-s{s}"/"model.pt" for s in (1, 2, 3)],
        c1mm=[args.reference_root/f"hls-c1mm-final-s{s}"/"model.pt" for s in (1, 2, 3, 4, 5)], prithvi=[])
    caches = {m: args.parent.parent/(args.parent.stem+"-"+m+".npy") for m in models}
    # Check every handle before invoking the existing cache-verifying reader.
    if any(not p.exists() or not p.with_name(p.name+".binding.json").exists() for p in caches.values()):
        raise ValueError("missing completed cache; inference forbidden")
    for model, paths in models.items():
        if model != "prithvi" and [sha256_file(p) for p in paths] != parent["model_hashes"][model]:
            raise ValueError("parent model identity differs")
    probabilities = {m: load_completed_prediction(args.data_root, records, paths, m, caches[m], args.prithvi_root)
        for m, paths in models.items()}
    _, reconstructed = counts_for(args.data_root, records, probabilities)
    verify_rows(records, parent["per_window"], reconstructed)
    groups = group_influence(parent["per_window"])
    missed_groups = set(parent["errors"]["models"]["reserve"]["missed_positive_groups"])
    missed_windows = set(parent["errors"]["models"]["reserve"]["missed_positive_windows"])
    windows = []
    for i, record in enumerate(records):
        pair, labels, valid = read_sample(args.data_root, record, "pair")
        pre, post = pair[:6], pair[6:]
        with rasterio.open(args.data_root/record["files"]["post_scl"]) as source:
            water = l2a_water(post, source.read(1))
        masks = diagnostic_masks(labels, valid, water, probabilities["reserve"][i], probabilities["c1mm"][i])
        indices = spectral_indices(pre, post)
        true = masks["labelled_burn"]
        item = dict(sample_id=record["sample_id"], group_id=record["group_id"],
            seed_area_ha=record["area_ha"], valid_fraction=record["valid_fraction"],
            pre_at=record["pre_at"], post_at=record["post_at"],
            days_after_interval=(datetime.fromisoformat(record["post_at"])-
                datetime.fromisoformat(record["ignition_interval"][1].replace("Z", "+00:00"))).total_seconds()/86400,
            spectra={name: spectral_summary(indices, mask) for name, mask in masks.items()},
            water_truth_pixels=int((true & water).sum()),
            probabilities_on_truth={m: dict(max_all=probability_max(p[i], true),
                max_without_water=probability_max(p[i], true & ~water)) for m, p in probabilities.items()},
            counts={m: parent["per_window"][i][m] for m in models},
            reserve_missed_window=record["sample_id"] in missed_windows,
            reserve_missed_group=record["group_id"] in missed_groups)
        windows.append(item)
    result = dict(spec_id="SPEC-106", parent_sha256=PARENT_SHA,
        manifest_sha256=sha256_file(args.manifest), recipe_sha256=sha256_file(recipe_file),
        cache_sha256={m: sha256_file(p) for m, p in caches.items()},
        all_counts_reconstructed=True, groups=groups, windows=windows,
        source_hashes={str(Path(__file__)): sha256_file(__file__),
            "src/comp/geographic_failure_report.py": sha256_file("src/comp/geographic_failure_report.py")},
        non_claims=["Consumed evaluation: descriptive diagnosis, not independent confirmation or causal proof.",
            "Leave-one-group-out is influence, not a new CI or authorization to exclude groups.",
            "No new model, field truth, reviewed negatives, real ignition dates, accuracy gain or goal completion."])
    content = json.dumps(result, indent=2, allow_nan=False)
    if len(content.encode()) >= 2*1024**2:
        raise ValueError("compact diagnostic output limit exceeded")
    with args.out.open("x") as stream:
        stream.write(content+"\n")
    print(json.dumps(dict(windows=len(windows), groups=len(groups), missed_windows=len(missed_windows),
        missed_groups=len(missed_groups), counts_reconstructed=True)), flush=True)


if __name__ == "__main__":
    main()
