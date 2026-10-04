"""SPEC-118: five-seed reference anchor, three learned corrections, all old populations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet import UNet
from scripts.forest_pair_train import read_sample
from scripts.report_geographic_failures import load_completed_prediction
from scripts.score_russian_transfer import counts_for
from scripts.score_forest_pair import binary_counts
from src.comp.accuracy_contract import bind_cache, canonical_hash, experiment_fingerprint, sha256_file, verify_cache
from src.comp.accuracy_artifacts import checkpoint_metadata, execution_sources, freeze_recipe, load_checkpoint
from src.comp.hls import tta8
from src.comp.l2a_finetuning import l2a_water
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi
from src.comp.recall_expansion import RecallExpansion, expand_probability
from src.comp.five_seed_additions import audit_all, validate_member, contribution, FIVE_BASE_SHAS as BASE_SHAS
from src.comp.temporal_adapter_contract import (BASE_MANIFEST_SHA, RUSSIAN_MANIFEST_SHA,
    fit_population, guard_output, read_seal)
from src.comp.temporal_adapter_evaluation import development_decision, pack_prediction, unpack_prediction

PARENT_SHAS = {
    "russian-transfer-v1-result": "242acac3cf2ccd30728e5919997895e7cf1d83de223a8802b06f3c4daaaa296d",
    "western-holdout-v1-result": "749699f3f6e01f0a0a6576afb496c891c4bc33c05ad1cca4087585ed5628cb2c",
    "reference-additions-v1-confirmation": "ec03c8313f2839cf72668f5bad27e9c01d26fc989d7db6ade71bb5a557085c16",
}


def water_for(root, record, post):
    if record.get("date_semantics") == "annual-interval-pair":
        with rasterio.open(root/record["files"]["post_scl"]) as source:
            return l2a_water(post, source.read(1))
    fmask, _ = evaluation_fmask(root, record)
    return water_ndwi(fmask, post[1], post[3])


def audit_models(args, fit, populations, seal):
    return audit_all(args, fit, populations, seal)


def expanded_predictions(args, records, anchor, paths, variant, cache):
    if len(paths) != 5:
        raise ValueError("complete uniform five-member prediction required")
    inputs = [args.data_root/r["files"][k] for r in records for k in sorted(r["files"])]
    bases = [args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)]
    models = paths+bases+[p.with_name(p.stem+".metadata.json") for p in paths]
    sources = execution_sources(Path(__file__), "src/comp/recall_expansion.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/score_equal_cost_additions.py", "src/comp/temporal_adapter_evaluation.py",
        "scripts/forest_pair_train.py", "src/comp/l2a_finetuning.py", "src/comp/terrain.py", "src/comp/severity_layers.py")
    signature = experiment_fingerprint(records, inputs, models, sources,
        dict(variant=variant, anchor_sha256=args.anchor_cache_sha, tta=8, threshold=.5,
            precision="bf16", format="little-packbits-final-valid-water", ensemble="C1-MM5+uniform-mean5-addition-union"))
    shape = (len(records), 512, 512)
    if cache.exists():
        verify_cache(cache, signature)
        return unpack_prediction(np.load(cache, allow_pickle=False), shape)
    corrections = np.zeros(shape, np.float32)
    for seed, (path, base_path) in enumerate(zip(paths, bases), 1):
        b = load_checkpoint(base_path, BASE_SHAS[seed-1], export_root=Path("research/tensor-exports"))
        base = UNet(6, classes=2, w=32, depth=7)
        base.load_state_dict(b["state"])
        mean, std = np.asarray(b["mean"], np.float32), np.asarray(b["std"], np.float32)
        del b
        adapter = load_checkpoint(path, sha256_file(path))
        validate_member(adapter, seed, variant, read_seal(args.runs/"recipe.json")["recipe_id"])
        if adapter["mean"] != mean.tolist() or adapter["std"] != std.tolist():
            raise ValueError("adapter changed initializer normalization")
        model = RecallExpansion(base, variant).cuda().eval()
        model.load_adapter_state(adapter["state"])
        del adapter
        for i, record in enumerate(records):
            image, _, _ = read_sample(args.data_root, record, "pair")
            x = torch.from_numpy(((image-np.tile(mean, 2)[:, None, None])/
                np.tile(std, 2)[:, None, None]).astype(np.float32))[None].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                correction = tta8(lambda a: model.forward_with_addition(a)[1][:, None], x)[0, 0]
            if not torch.isfinite(correction).all():
                raise ValueError("nonfinite correction")
            corrections[i] += contribution(correction, len(paths)).cpu().numpy()
        del model, base
    probability = expand_probability(torch.from_numpy(np.array(anchor, copy=True)), torch.from_numpy(corrections)).numpy()
    prediction = np.zeros(shape, bool)
    for i, record in enumerate(records):
        post, _, valid = read_sample(args.data_root, record, "post")
        observed = valid & ~water_for(args.data_root, record, post)
        prediction[i] = (probability[i] >= .5) & observed
        if ((anchor[i] >= .5) & observed & ~prediction[i]).any():
            raise ValueError("monotonic expansion removed reference-positive pixels")
    guard_output(args.runs)
    with cache.open("xb") as stream:
        np.save(stream, pack_prediction(prediction))
    bind_cache(cache, signature)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "runs", "reference-root", "prithvi-root", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable development result already exists")
    seal = read_seal(args.runs/"recipe.json")
    for name, digest in PARENT_SHAS.items():
        if sha256_file(Path("research")/(name+".json")) != digest:
            raise ValueError("completed parent changed")
    for name, digest in (("forest-pairs/manifest.json", BASE_MANIFEST_SHA), ("russia/manifest.json", RUSSIAN_MANIFEST_SHA)):
        if sha256_file(args.data_root/name) != digest:
            raise ValueError("frozen source population changed")
    base = json.loads((args.data_root/"forest-pairs/manifest.json").read_text())["records"]
    russian = json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    west = json.loads((args.data_root/"western-holdout-v1/manifest.json").read_text())
    old = json.loads(Path("research/russian-transfer-v1-result.json").read_text())
    western = json.loads(Path("research/western-holdout-v1-result.json").read_text())
    calibration_file = Path("research/reference-additions-v1-confirmation.json")
    calibration = json.loads(calibration_file.read_text())
    if (sha256_file(args.data_root/"western-holdout-v1/manifest.json") != western["manifest_sha256"]
            or calibration["model_hashes"] != old["model_hashes"]):
        raise ValueError("geographic/calibration parent differs")
    populations = dict(russia_eval=[r for r in russian if r["partition"] == "evaluation"],
        russia_cal=[r for r in russian if r["partition"] == "calibration"], western=west["records"],
        ems=[r for r in base if r["partition"] == "evaluation" and r["dataset"] == "ems"],
        floga=[r for r in base if r["partition"] == "evaluation" and r["dataset"] == "floga"])
    if {k: len(v) for k, v in populations.items()} != dict(russia_eval=77, russia_cal=61, western=96, ems=60, floga=81):
        raise ValueError("development population incomplete")
    paths, metadata, summaries = audit_models(args, fit_population(base, russian), populations, seal)
    reference_models = dict(reserve=[Path("research/russian-transfer-v1")/f"reserve-s{s}/model.pt" for s in (1, 2, 3)],
        c1mm=[args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)], prithvi=[])
    for name in ("reserve", "c1mm"):
        if [sha256_file(p) for p in reference_models[name]] != old["model_hashes"][name]:
            raise ValueError("original comparator weights changed")
    sources = execution_sources(Path(__file__), "src/comp/recall_expansion.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/score_equal_cost_additions.py", "src/comp/temporal_adapter_evaluation.py",
        "scripts/report_geographic_failures.py", "scripts/score_russian_transfer.py",
        "scripts/forest_pair_train.py", "src/comp/l2a_finetuning.py", "src/comp/terrain.py", "src/comp/severity_layers.py",
        "src/comp/temporal_adapter_contract.py", "scripts/queues/queue_score_five_seed_additions.sh")
    recipe = freeze_recipe(args.out.with_suffix(".recipe.json"), sources, dict(training_recipe=seal["recipe_id"],
        adapters={v: [sha256_file(p) for p in ps] for v, ps in paths.items()},
        parents={"calibration": sha256_file(calibration_file), **PARENT_SHAS},
        references=old["model_hashes"],
        prithvi_weights=sha256_file(args.prithvi_root/"Prithvi_EO_V2_300M_BurnScars.pt"),
        prithvi_config=sha256_file(args.prithvi_root/"burn_scars_config.yaml"),
        ordered_ids={k: [r["sample_id"] for r in rs] for k, rs in populations.items()},
        tta=8, threshold=.5, bootstrap=2000, seed=20261003))
    datasets = {}
    for name, records in populations.items():
        for record in records:
            for key, relative in record["files"].items():
                if sha256_file(args.data_root/relative) != record["hashes"][key]:
                    raise ValueError("development input changed")
        stem = {"russia_eval": "russian-transfer-v1-result", "russia_cal": "reference-additions-v1-confirmation",
            "western": "western-holdout-v1-result", "ems": "russian-transfer-v1-result-ems", "floga": "russian-transfer-v1-result-floga"}[name]
        wanted = ["reserve", "c1mm"]+(["prithvi"] if name.startswith("russia") or name == "western" else [])
        references = {m: load_completed_prediction(args.data_root, records, reference_models[m], m,
            Path("research")/(stem+"-"+m+".npy"), args.prithvi_root) for m in wanted}
        _, rows = counts_for(args.data_root, records, references)
        parent_rows = {"russia_eval": old.get("per_window"), "russia_cal": calibration.get("per_window"), "western": western.get("per_window")}.get(name)
        if parent_rows is not None:
            if len(parent_rows) != len(rows):
                raise ValueError("original comparator population length differs")
            for got, expected in zip(rows, parent_rows):
                if any(got[k] != expected[k] for k in ("sample_id", "group_id", "valid_pixels", *wanted)):
                    raise ValueError("original comparator counts differ")
        args.anchor_cache_sha = sha256_file(Path("research")/(stem+"-c1mm.npy"))
        predicted = {v: expanded_predictions(args, records, references["c1mm"], ps, v,
            args.runs/(name+"-"+v+".packed.npy")) for v, ps in paths.items()}
        for i, (row, record) in enumerate(zip(rows, records)):
            _, label, valid = read_sample(args.data_root, record, "post")
            row.update({v: binary_counts(label == 1, values[i], valid) for v, values in predicted.items()})
            row["total_pixels"] = record["grid"]["width"]*record["grid"]["height"]
        datasets[name] = rows
        print("SCORED", name, len(rows), flush=True)
    result = dict(spec_id="SPEC-118", **development_decision(datasets), per_window=datasets,
        training_recipe_id=seal["recipe_id"], scoring_recipe_id=recipe,
        adapter_hashes={v: [sha256_file(p) for p in ps] for v, ps in paths.items()},
        matched_epoch_traces=True, full_fit_exposure=True,
        non_claims=["Consumed development populations; eligibility is not independent confirmation or promotion.",
            "PANGAEA background is published coverage, not reviewed absence of fire; spatial groups are not dated fires.",
            "No field truth, severity/L2A useful gain, global ranking or original goal completion."])
    guard_output(args.runs)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k in ("eligible", "guards")}), flush=True)


if __name__ == "__main__":
    main()
