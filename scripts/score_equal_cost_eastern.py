"""SPEC-116: new geography, frozen eligible models, same source pixels and full guards."""
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_equal_cost_eastern import prepare_binding
from scripts.prepare_russian_windows import interval_pair_valid
from scripts.score_equal_cost_additions import expanded_predictions, audit_models
from scripts.score_russian_transfer import predict, counts_for
from scripts.forest_pair_train import read_sample
from scripts.score_forest_pair import binary_counts
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.accuracy_contract import canonical_hash, paired_group_comparison, sha256_file, validate_manifest
from src.comp.equal_cost_eastern import (DATA_BUDGET, SELECTION_SHA, cache_guard, require_admission, confirmation_decision)
from src.comp.l2a_finetuning import l2a_valid
from src.comp.russian_error_report import summarize
from src.comp.temporal_adapter_contract import fit_population, read_seal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "selection", "candidate", "runs", "reference-root", "prithvi-root", "manifest", "out"):
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    if (args.out.resolve() != Path("research/eastern-confirmation-v2/result.json").resolve()
            or args.manifest.resolve() != (args.data_root/"eastern-confirmation-v2/manifest.json").resolve()
            or args.out.exists()):
        raise ValueError("immutable fresh confirmation result exists")
    selection, binding = prepare_binding(args)
    payload = json.loads(args.manifest.read_text())
    require_admission(payload)
    records = payload["records"]
    counts = validate_manifest(records)
    if counts != payload["counts"]:
        raise ValueError("admission population differs")
    bound = json.loads(args.manifest.with_name("binding.json").read_text())
    if (bound["configuration"] != binding or bound["recipe_id"] != payload["binding_recipe_id"]
            or canonical_hash({k: v for k, v in bound.items() if k != "recipe_id"}) != bound["recipe_id"]):
        raise ValueError("pre-image candidate binding changed")
    for name, digest in bound["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("collected source recipe changed")
    used = sum(p.stat().st_size for p in args.manifest.parent.rglob("*") if p.is_file())
    if used > DATA_BUDGET:
        raise ValueError("source imagery budget exceeded")
    selected = {r["sample_id"]: r for r in selection["records"]}
    for record in records:
        original = selected.get(record["sample_id"])
        if original is None or any(record[k] != original[k] for k in ("grid", "bbox_wgs84", "group_id", "partition", "center")):
            raise ValueError("collected frame not in frozen population")
        interval_pair_valid(record)
        arrays = {}
        for key, relative in record["files"].items():
            if sha256_file(args.data_root/relative) != record["hashes"][key]:
                raise ValueError("collected source bytes changed")
            with rasterio.open(args.data_root/relative) as source:
                if (str(source.crs) != record["grid"]["crs"] or source.shape != (512, 512)
                        or list(source.transform)[:6] != record["grid"]["transform"]):
                    raise ValueError("collected source grid differs")
                arrays[key] = source.read() if key in ("pre", "post") else source.read(1)
        polygons = gpd.read_file("zip://"+str(args.data_root/"russia/burned_area_2020_topology.zip"), bbox=tuple(record["bbox_wgs84"]))
        if str(polygons.crs) != "EPSG:4326":
            raise ValueError("published annotation CRS differs")
        polygons = polygons[polygons.geometry.notna() & ~polygons.geometry.is_empty & polygons.geometry.is_valid].to_crs(record["grid"]["crs"])
        truth = rasterize([(g, 1) for g in polygons.geometry], out_shape=(512, 512),
            transform=rasterio.Affine(*record["grid"]["transform"]), fill=0, dtype="int16")
        expected = l2a_valid(arrays["pre"], arrays["pre_scl"], truth) & l2a_valid(arrays["post"], arrays["post_scl"], truth)
        if (not np.array_equal(expected, arrays["valid"].astype(bool))
                or not np.array_equal(np.where(expected, truth, -1), arrays["mask"])
                or expected.mean() < .5 or ((truth == 1) & expected).sum() < 9):
            raise ValueError("strict source supervision/admission differs")
    base = json.loads((args.data_root/"forest-pairs/manifest.json").read_text())["records"]
    russian = json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    seal = read_seal(args.runs/"recipe.json")
    paths, _, _ = audit_models(args, fit_population(base, russian), dict(fresh=records), seal)
    refs = dict(c1mm=[args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)],
        reserve=[Path("research/russian-transfer-v1")/f"reserve-s{s}/model.pt" for s in (1, 2, 3)], prithvi=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sources = execution_sources(Path(__file__), "scripts/collect_equal_cost_eastern.py", "src/comp/equal_cost_eastern.py", "src/comp/eastern_confirmation.py", "src/comp/equal_cost_additions.py",
        "scripts/score_equal_cost_additions.py", "src/comp/recall_expansion.py", "scripts/score_russian_transfer.py",
        "scripts/forest_pair_train.py", "src/comp/l2a_finetuning.py", "src/comp/terrain.py", "src/comp/severity_layers.py")
    scoring = freeze_recipe(args.out.with_suffix(".recipe.json"), sources,
        dict(binding=binding, source_manifest=sha256_file(args.manifest), threshold=.5, tta=8, bootstrap=2000, seed=20261003))
    probabilities = {}
    for name, models in refs.items():
        cache = args.out.parent/(name+".npy")
        if not cache.exists():
            cache_guard(args.out.parent, len(records)*512*512*4+1024*1024)
        probabilities[name] = predict(args.data_root, records, models, name, cache, args.prithvi_root)
    _, rows = counts_for(args.data_root, records, probabilities)
    args.anchor_cache_sha = sha256_file(args.out.parent/"c1mm.npy")
    predictions = {}
    for variant, models in paths.items():
        cache = args.out.parent/(variant+".packed.npy")
        if not cache.exists():
            cache_guard(args.out.parent, len(records)*512*64+1024*1024)
        predictions[variant] = expanded_predictions(args, records, probabilities["c1mm"], models, variant, cache)
    for i, (row, record) in enumerate(zip(rows, records)):
        _, truth, valid = read_sample(args.data_root, record, "post")
        row.update({v: binary_counts(truth == 1, masks[i], valid) for v, masks in predictions.items()})
        row["total_pixels"] = record["grid"]["width"]*record["grid"]["height"]
    errors = summarize(rows, ["pair", "post", "c1mm", "reserve", "prithvi"])
    comparisons = {m: paired_group_comparison([r["group_id"] for r in rows], [r["pair"] for r in rows], [r[m] for r in rows])
        for m in ("post", "c1mm", "reserve", "prithvi")}
    missed = {m: len(r["missed_positive_groups"]) for m, r in errors["models"].items()}
    result = dict(spec_id="SPEC-116", accepted=confirmation_decision(comparisons, missed),
        comparisons=comparisons, missed_positive_groups=missed, errors=errors, per_window=rows,
        binding=binding, source_manifest_sha256=sha256_file(args.manifest), scoring_recipe_id=scoring,
        non_claims=["New recorded-history-disjoint selected geography, shared PANGAEA method/spring2020.",
            "Spatial groups are not physical fire IDs; background not reviewed absence of fire.",
            "No product promotion, field truth, global ranking, grade/L2A success or original goal completion."])
    cache_guard(args.out.parent, 1024*1024)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k in ("accepted", "comparisons", "missed_positive_groups")}), flush=True)


if __name__ == "__main__":
    main()
