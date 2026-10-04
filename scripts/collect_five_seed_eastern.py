"""SPEC-119: eligibility and artifact checks precede all new catalogue/data work."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_artifacts import freeze_recipe
from src.comp.accuracy_contract import canonical_hash, sha256_file, validate_manifest
from src.comp.five_seed_eastern import SELECTION_SHA, TRAINING_RECIPE, StorageBudget, require_eligible
from src.comp.eastern_confirmation_plan import REGISTRIES, verify_histories
from src.comp.temporal_adapter_contract import BASE_MANIFEST_SHA, RUSSIAN_MANIFEST_SHA, fit_population, read_seal
from src.comp.temporal_adapter_evaluation import development_decision
from scripts.prepare_russian_windows import choose, interval_pair_valid
from scripts.prepare_western_holdout import SOURCE_SHA
from scripts.score_five_seed_additions import audit_models, expanded_predictions
from scripts.report_geographic_failures import load_completed_prediction
from scripts.score_russian_transfer import counts_for
from scripts.score_forest_pair import binary_counts
from scripts.forest_pair_train import read_sample


def authenticate_counts(args, candidate, paths):
    base = json.loads((args.data_root/"forest-pairs/manifest.json").read_text())["records"]
    russian = json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    western = json.loads((args.data_root/"western-holdout-v1/manifest.json").read_text())["records"]
    populations = dict(russia_eval=[r for r in russian if r["partition"] == "evaluation"],
        russia_cal=[r for r in russian if r["partition"] == "calibration"], western=western,
        ems=[r for r in base if r["partition"] == "evaluation" and r["dataset"] == "ems"],
        floga=[r for r in base if r["partition"] == "evaluation" and r["dataset"] == "floga"])
    if set(candidate["per_window"]) != set(populations):
        raise ValueError("candidate development population changed")
    stems = dict(russia_eval="russian-transfer-v1-result", russia_cal="reference-additions-v1-confirmation",
        western="western-holdout-v1-result", ems="russian-transfer-v1-result-ems", floga="russian-transfer-v1-result-floga")
    models = dict(c1mm=[args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)],
        reserve=[Path("research/russian-transfer-v1")/f"reserve-s{s}/model.pt" for s in (1, 2, 3)], prithvi=[])
    for name, records in populations.items():
        wanted = ["reserve", "c1mm"]+(["prithvi"] if name in ("russia_eval", "russia_cal", "western") else [])
        for record in records:
            if record.get("date_semantics") != "annual-interval-pair":
                legacy = args.data_root/record["files"]["post"].replace("_merged.tif", ".fmask.tif")
                own = args.data_root/"severity-fmask"/(canonical_hash(dict(item=record["post_item"], grid=record["grid"]))+".tif")
                if not legacy.exists() and not (own.exists() and own.with_suffix(".json").exists()):
                    raise ValueError("completed observation mask missing; no catalogue access allowed")
        arrays = {m: load_completed_prediction(args.data_root, records, models[m], m,
            Path("research")/(stems[name]+"-"+m+".npy"), args.prithvi_root) for m in wanted}
        _, rows = counts_for(args.data_root, records, arrays)
        args.anchor_cache_sha = sha256_file(Path("research")/(stems[name]+"-c1mm.npy"))
        predictions = {}
        for variant, checkpoints in paths.items():
            cache = args.runs/(name+"-"+variant+".packed.npy")
            if not cache.exists() or not cache.with_name(cache.name+".binding.json").exists():
                raise ValueError("candidate cache missing; no inference allowed")
            predictions[variant] = expanded_predictions(args, records, arrays["c1mm"], checkpoints, variant, cache)
        for i, (row, record) in enumerate(zip(rows, records)):
            _, truth, valid = read_sample(args.data_root, record, "post")
            row.update({v: binary_counts(truth == 1, mask[i], valid) for v, mask in predictions.items()})
            row["total_pixels"] = record["grid"]["height"]*record["grid"]["width"]
        if rows != candidate["per_window"][name]:
            raise ValueError("candidate counts do not represent authenticated cached predictions")


def prepare_binding(args):
    candidate = json.loads(args.candidate.read_text())
    require_eligible(candidate)  # No directory/catalogue work precedes this.
    computed = development_decision(candidate["per_window"])
    if any(candidate[k] != v for k, v in computed.items()):
        raise ValueError("candidate eligibility/counts were altered")
    if sha256_file(args.selection) != SELECTION_SHA:
        raise ValueError("preregistered unused population changed")
    selection = json.loads(args.selection.read_text())
    if (selection["spec_id"] != "SPEC-112" or not selection["ready_metadata_population"]
            or selection["counts"]["samples"] != 72 or selection["counts"]["groups"] != 43):
        raise ValueError("wrong metadata population")
    histories = {name: (sha256_file(args.data_root/name), json.loads((args.data_root/name).read_text())) for name in REGISTRIES}
    if verify_histories(histories) != selection["exclusion_audit"]:
        raise ValueError("exclusion registry audit changed")
    if sha256_file(args.data_root/"russia/burned_area_2020_topology.zip") != SOURCE_SHA:
        raise ValueError("source annotations changed")
    for name, digest in selection["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("metadata selection source changed")
    seal = read_seal(args.runs/"recipe.json")
    if (seal["recipe_id"] != candidate["training_recipe_id"] or seal["recipe_id"] != TRAINING_RECIPE
            or seal["configuration"].get("loss_id") != "equal-ce-dice-v1"
            or seal["configuration"].get("ce_weights") != [1, 1]
            or seal["configuration"].get("scoring") != "C1-MM5+uniform-mean5-probability-union"):
        raise ValueError("candidate training seal differs")
    scoring = json.loads(args.candidate.with_suffix(".recipe.json").read_text())
    if (scoring["recipe_id"] != candidate["scoring_recipe_id"]
            or canonical_hash({k: v for k, v in scoring.items() if k != "recipe_id"}) != scoring["recipe_id"]):
        raise ValueError("candidate scoring seal differs")
    expected_settings = dict(tta=8, threshold=.5, bootstrap=2000, seed=20261003)
    if any(scoring["configuration"].get(k) != v for k, v in expected_settings.items()):
        raise ValueError("candidate scoring settings differ")
    for name, digest in scoring["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("candidate scoring source changed")
    for relative, digest in (("forest-pairs/manifest.json", BASE_MANIFEST_SHA), ("russia/manifest.json", RUSSIAN_MANIFEST_SHA)):
        if sha256_file(args.data_root/relative) != digest:
            raise ValueError("candidate fitted population changed")
    base = json.loads((args.data_root/"forest-pairs/manifest.json").read_text())["records"]
    russian = json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    fit = fit_population(base, russian)
    paths, _, _ = audit_models(args, fit, dict(eastern=selection["records"]), seal)
    hashes = {v: [sha256_file(p) for p in ps] for v, ps in paths.items()}
    if hashes != candidate["adapter_hashes"]:
        raise ValueError("eligible candidate adapters changed")
    if {r["sample_id"].split(":", 1)[-1] for r in fit} & {r["sample_id"].split(":", 1)[-1] for r in selection["records"]}:
        raise ValueError("selected source geometry was fitted")
    refs = scoring["configuration"]["references"]
    for seed, digest in enumerate(refs["c1mm"], 1):
        if sha256_file(args.reference_root/f"hls-c1mm-final-s{seed}/model.pt") != digest:
            raise ValueError("reference C1-MM changed")
    for seed, digest in enumerate(refs["reserve"], 1):
        if sha256_file(Path("research/russian-transfer-v1")/f"reserve-s{seed}/model.pt") != digest:
            raise ValueError("original reserve changed")
    for filename, key in (("Prithvi_EO_V2_300M_BurnScars.pt", "prithvi_weights"), ("burn_scars_config.yaml", "prithvi_config")):
        if sha256_file(args.prithvi_root/filename) != scoring["configuration"][key]:
            raise ValueError("official comparator changed")
    authenticate_counts(args, candidate, paths)
    return selection, dict(candidate_sha256=sha256_file(args.candidate), selection_sha256=SELECTION_SHA,
        loss_id="equal-ce-dice-v1", candidate_spec_id="SPEC-118", members=5, training_recipe=seal["recipe_id"], scoring_recipe=scoring["recipe_id"], adapters=hashes,
        references=refs, prithvi_weights=scoring["configuration"]["prithvi_weights"],
        prithvi_config=scoring["configuration"]["prithvi_config"], source_zip=SOURCE_SHA)


def collect_window(root, out, record, budget):
    budget.reserve()
    try:
        catalogue = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace)
        post, xp, sp, vp, ep = choose(catalogue, record, 2020)
        tile = post.properties["s2:mgrs_tile"]
        pre, xb, sb, vb, eb = choose(catalogue, record, 2019, tile=tile, target_date=post.datetime.replace(year=2019))
        valid = vp & vb
        if valid.mean() < .5:
            raise ValueError("insufficient common observation")
        source = gpd.read_file("zip://"+str(root/"russia/burned_area_2020_topology.zip"), bbox=tuple(record["bbox_wgs84"]))
        if str(source.crs) != "EPSG:4326":
            raise ValueError("annotation CRS differs")
        source = source[source.geometry.notna() & ~source.geometry.is_empty & source.geometry.is_valid].to_crs(record["grid"]["crs"])
        transform = rasterio.Affine(*record["grid"]["transform"])
        truth = rasterize([(g, 1) for g in source.geometry], out_shape=(512, 512), transform=transform, fill=0, dtype="int16")
        if ((truth == 1) & valid).sum() < 9:
            raise ValueError("insufficient retained published burn")
        values = dict(pre=(np.where(vb[None], xb, -9999), "float32", -9999),
            post=(np.where(vp[None], xp, -9999), "float32", -9999), pre_scl=(sb[None], "uint8", 0),
            post_scl=(sp[None], "uint8", 0), valid=(valid.astype(np.uint8)[None], "uint8", 0),
            mask=(np.where(valid, truth, -1)[None], "int16", -1))
        files = {}
        for key, (array, dtype, nodata) in values.items():
            path = out/(record["sample_id"].replace(":", "_")+"."+key+".tif")
            if path.exists():
                raise ValueError("immutable partial data file exists")
            with budget.lock:
                budget.check_write()
                with rasterio.open(path, "w", driver="GTiff", height=512, width=512, count=len(array), dtype=dtype,
                    crs=record["grid"]["crs"], transform=transform, compress="deflate", nodata=nodata) as target:
                    target.write(array.astype(dtype))
            files[key] = str(path.relative_to(root))
        result = dict(record, files=files, hashes={k: sha256_file(root/p) for k, p in files.items()},
            ignition_interval=["2020-01-01T00:00:00Z", "2020-05-16T00:00:00Z"],
            pre_at=pre.datetime.isoformat(), post_at=post.datetime.isoformat(), pre_item=pre.id, post_item=post.id,
            mgrs=tile, pre_mgrs=pre.properties["s2:mgrs_tile"], pre_grid=record["grid"], encodings=dict(pre=eb, post=ep),
            date_semantics="annual-interval-pair", source_provenance="participatory_perimeter",
            valid_fraction=float(valid.mean()), fire_pixels=int(((truth == 1) & valid).sum()))
        interval_pair_valid(result)
        with budget.lock:
            budget.check_write()
            with (out/(record["sample_id"].replace(":", "_")+".json")).open("x") as stream:
                json.dump(result, stream, indent=2)
        print("COLLECTED", record["sample_id"], flush=True)
        return result
    finally:
        budget.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "selection", "candidate", "runs", "reference-root", "prithvi-root", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    args = p.parse_args()
    selection, binding = prepare_binding(args)
    root, out = args.data_root.resolve(), args.out.resolve()
    if out != root/"eastern-confirmation-v3" or out.exists():
        raise ValueError("fresh own output directory required")
    out.mkdir()
    budget = StorageBudget(out)
    budget.check_write()
    sources = (str(Path(__file__)), "src/comp/five_seed_eastern.py", "src/comp/eastern_confirmation.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/prepare_russian_windows.py",
        "scripts/score_five_seed_additions.py", "src/comp/l2a_finetuning.py", "src/comp/eastern_confirmation_plan.py",
        "scripts/queues/queue_five_seed_eastern.sh", "scripts/queues/queue_score_five_seed_eastern.sh")
    seal = freeze_recipe(out/"binding.json", sources, binding)
    def run(record):
        try:
            return collect_window(root, out, record, budget), None
        except Exception as error:
            return None, dict(sample_id=record["sample_id"], reason=type(error).__name__)  # No signed URL messages.
    with ThreadPoolExecutor(2) as pool:
        outcomes = list(pool.map(run, selection["records"]))
    records = sorted((r for r, e in outcomes if r), key=lambda r: r["sample_id"])
    counts = validate_manifest(records) if records else dict(samples=0, groups=0)
    ready = counts["samples"] >= 50 and counts["groups"] >= 20
    result = dict(spec_id="SPEC-119", records=records, counts=counts, failures=[e for r, e in outcomes if e],
        selection_sha256=SELECTION_SHA, binding_recipe_id=seal, ready_for_evaluation=bool(ready), qc_probe=False,
        storage_bytes=budget.used(), non_claims=selection["non_claims"])
    budget.check_write()
    with (out/"manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k not in ("records", "failures")}), flush=True)
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
