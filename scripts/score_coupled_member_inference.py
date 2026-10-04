"""SPEC-121: five-seed reference anchor, three learned corrections, all old populations."""
import argparse
import json
from pathlib import Path
from argparse import Namespace
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
from src.comp.accuracy_contract import bind_cache, canonical_hash, experiment_fingerprint, sha256_file, verify_cache, paired_group_comparison
from src.comp.accuracy_artifacts import checkpoint_metadata, execution_sources, freeze_recipe, load_checkpoint
from src.comp.hls import tta8
from src.comp.l2a_finetuning import l2a_water
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi
from src.comp.recall_expansion import RecallExpansion
from src.comp.coupled_member_inference import member_delta, anchored_probability, guard_output, save_mask, load_mask
from src.comp.eastern_confirmation import confirmation_decision
from src.comp.russian_error_report import summarize
from src.comp.five_seed_additions import audit_all, validate_member, contribution, FIVE_BASE_SHAS as BASE_SHAS
from src.comp.temporal_adapter_contract import (BASE_MANIFEST_SHA, RUSSIAN_MANIFEST_SHA,
    fit_population, read_seal)
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
    return audit_all(Namespace(data_root=args.data_root,runs=args.members_root), fit, populations, seal)


def expanded_predictions(args, records, anchor, paths, variant, cache):
    if len(paths) != 5:
        raise ValueError("complete uniform five-member prediction required")
    inputs = [args.data_root/r["files"][k] for r in records for k in sorted(r["files"])]
    bases = [args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)]
    models = paths+bases+[p.with_name(p.stem+".metadata.json") for p in paths]
    sources = execution_sources(Path(__file__), "src/comp/recall_expansion.py", "src/comp/coupled_member_inference.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/score_equal_cost_additions.py", "src/comp/temporal_adapter_evaluation.py",
        "scripts/forest_pair_train.py", "src/comp/l2a_finetuning.py", "src/comp/terrain.py", "src/comp/severity_layers.py")
    signature = experiment_fingerprint(records, inputs, models, sources,
        dict(variant=variant, anchor_sha256=args.anchor_cache_sha, tta=8, threshold=.5,
            precision="bf16", format="little-packbits-deflate-final-valid-water", ensemble="C1-MM5+uniform-mean5-member-residual"))
    shape = (len(records), 512, 512)
    if cache.exists():
        verify_cache(cache, signature)
        return load_mask(cache,signature,shape)
    corrections = np.zeros(shape, np.float32)
    reproduced_anchor = np.zeros(shape,np.float32)
    for seed, (path, base_path) in enumerate(zip(paths, bases), 1):
        b = load_checkpoint(base_path, BASE_SHAS[seed-1], export_root=Path("research/tensor-exports"))
        base = UNet(6, classes=2, w=32, depth=7)
        base.load_state_dict(b["state"])
        reference_mean, reference_std = np.asarray(b["mean"]), np.asarray(b["std"])
        mean, std = np.asarray(b["mean"], np.float32), np.asarray(b["std"], np.float32)
        del b
        adapter = load_checkpoint(path, sha256_file(path))
        validate_member(adapter, seed, variant, read_seal(args.members_root/"recipe.json")["recipe_id"])
        if adapter["mean"] != mean.tolist() or adapter["std"] != std.tolist():
            raise ValueError("adapter changed initializer normalization")
        model = RecallExpansion(base, variant).cuda().eval()
        model.load_adapter_state(adapter["state"])
        del adapter
        for i, record in enumerate(records):
            image, _, _ = read_sample(args.data_root, record, "pair")
            x = torch.from_numpy(((image-np.tile(mean, 2)[:, None, None])/
                np.tile(std, 2)[:, None, None]).astype(np.float32))[None].cuda()
            ref_x=torch.from_numpy(((image[6:]-reference_mean[:,None,None])/reference_std[:,None,None]).astype(np.float32))[None].cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                member_probability=tta8(model.base,ref_x).softmax(1)[0,1]
                correction = tta8(lambda a: model.forward_with_addition(a)[1][:, None], x)[0, 0]
            if not torch.isfinite(correction).all():
                raise ValueError("nonfinite correction")
            corrections[i] += member_delta(member_probability,correction,len(paths)).cpu().numpy()
            reproduced_anchor[i] += member_probability.float().cpu().numpy()/5
        del model, base
    probability = anchored_probability(anchor,reproduced_anchor,corrections)
    prediction = np.zeros(shape, bool)
    for i, record in enumerate(records):
        post, _, valid = read_sample(args.data_root, record, "post")
        observed = valid & ~water_for(args.data_root, record, post)
        prediction[i] = (probability[i] >= .5) & observed
        if ((anchor[i] >= .5) & observed & ~prediction[i]).any():
            raise ValueError("monotonic expansion removed reference-positive pixels")
    guard_output(args.runs)
    save_mask(cache,prediction,signature)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "runs", "members-root", "reference-root", "prithvi-root", "out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable development result already exists")
    seal = read_seal(args.members_root/"recipe.json")
    for parent,digest in ((args.members_root/"development.json","fb6aa7f3fdd8a4f4d24dc4c04e8a9b33ebdd41117927ef24adff5288b907b851"),(Path("research/eastern-confirmation-v3/result.json"),"5b0d13886122636c4f7bfd88c14b1ed2b4406548a47adcd9b51ac315fe5d3c4b")):
        if sha256_file(parent)!=digest:raise ValueError("immutable candidate/source parent changed")
    args.runs.mkdir(parents=True,exist_ok=True)
    for name, digest in PARENT_SHAS.items():
        if sha256_file(Path("research")/(name+".json")) != digest:
            raise ValueError("completed parent changed")
    for name, digest in (("forest-pairs/manifest.json", BASE_MANIFEST_SHA), ("russia/manifest.json", RUSSIAN_MANIFEST_SHA)):
        if sha256_file(args.data_root/name) != digest:
            raise ValueError("frozen source population changed")
    base = json.loads((args.data_root/"forest-pairs/manifest.json").read_text())["records"]
    russian = json.loads((args.data_root/"russia/manifest.json").read_text())["records"]
    west = json.loads((args.data_root/"western-holdout-v1/manifest.json").read_text())
    parent118=json.loads((args.members_root/"development.json").read_text())
    eastern_parent=json.loads(Path("research/eastern-confirmation-v3/result.json").read_text())
    east_manifest=args.data_root/"eastern-confirmation-v3/manifest.json"
    if sha256_file(east_manifest)!=eastern_parent["source_manifest_sha256"]:raise ValueError("consumed eastern inputs changed")
    eastern=json.loads(east_manifest.read_text())["records"]
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
    populations["eastern_consumed"]=eastern
    if len(eastern)!=69:raise ValueError("consumed eastern population incomplete")
    paths, metadata, summaries = audit_models(args, fit_population(base, russian), populations, seal)
    reference_models = dict(reserve=[Path("research/russian-transfer-v1")/f"reserve-s{s}/model.pt" for s in (1, 2, 3)],
        c1mm=[args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)], prithvi=[])
    for name in ("reserve", "c1mm"):
        if [sha256_file(p) for p in reference_models[name]] != old["model_hashes"][name]:
            raise ValueError("original comparator weights changed")
    sources = execution_sources(Path(__file__), "src/comp/recall_expansion.py", "src/comp/coupled_member_inference.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/score_equal_cost_additions.py", "src/comp/temporal_adapter_evaluation.py",
        "scripts/report_geographic_failures.py", "scripts/score_russian_transfer.py",
        "scripts/forest_pair_train.py", "src/comp/l2a_finetuning.py", "src/comp/terrain.py", "src/comp/severity_layers.py",
        "src/comp/temporal_adapter_contract.py", "scripts/queues/queue_coupled_member_inference.sh")
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
            "western": "western-holdout-v1-result", "ems": "russian-transfer-v1-result-ems", "floga": "russian-transfer-v1-result-floga", "eastern_consumed":""}[name]
        def reference_cache(model):
            return Path("research/eastern-confirmation-v3")/(model+".npy") if name=="eastern_consumed" else Path("research")/(stem+"-"+model+".npy")
        wanted = ["reserve", "c1mm"]+(["prithvi"] if name.startswith("russia") or name in ("western","eastern_consumed") else [])
        references = {m: load_completed_prediction(args.data_root, records, reference_models[m], m,
            reference_cache(m), args.prithvi_root) for m in wanted}
        _, rows = counts_for(args.data_root, records, references)
        parent_rows = {"russia_eval": old.get("per_window"), "russia_cal": calibration.get("per_window"), "western": western.get("per_window"), "eastern_consumed":eastern_parent["per_window"]}.get(name)
        if parent_rows is not None:
            if len(parent_rows) != len(rows):
                raise ValueError("original comparator population length differs")
            for got, expected in zip(rows, parent_rows):
                if any(got[k] != expected[k] for k in ("sample_id", "group_id", "valid_pixels", *wanted)):
                    raise ValueError("original comparator counts differ")
        args.anchor_cache_sha = sha256_file(reference_cache("c1mm"))
        predicted = {v: expanded_predictions(args, records, references["c1mm"], ps, v,
            args.runs/(name+"-"+v+".packed.npz")) for v, ps in paths.items()}
        for i, (row, record) in enumerate(zip(rows, records)):
            _, label, valid = read_sample(args.data_root, record, "post")
            row.update({v: binary_counts(label == 1, values[i], valid) for v, values in predicted.items()})
            row["total_pixels"] = record["grid"]["width"]*record["grid"]["height"]
        datasets[name] = rows
        print("SCORED", name, len(rows), flush=True)
    old_datasets={k:v for k,v in datasets.items() if k!="eastern_consumed"}
    decision=development_decision(old_datasets)
    pooled=[r for name in ("russia_eval","russia_cal","western") for r in datasets[name]]
    prior=[r for name in ("russia_eval","russia_cal","western") for r in parent118["per_window"][name]]
    useful=paired_group_comparison([r["group_id"]for r in pooled],[r["pair"]for r in pooled],[r["pair"]for r in prior])
    east_rows=datasets["eastern_consumed"];east_errors=summarize(east_rows,["pair","post","c1mm","reserve","prithvi"])
    east_comparisons={k:paired_group_comparison([r["group_id"]for r in east_rows],[r["pair"]for r in east_rows],[r[k]for r in east_rows])for k in ("post","c1mm","reserve","prithvi")}
    east_missed={k:len(v["missed_positive_groups"])for k,v in east_errors["models"].items()}
    east_accepted=confirmation_decision(east_comparisons,east_missed)
    accepted=decision["eligible"] and useful["delta"]>=.005 and useful["ci95"][0]>0 and east_accepted
    result = dict(spec_id="SPEC-121", **decision, accepted=bool(accepted), parent_gain=useful,
        eastern_consumed=dict(accepted=east_accepted,comparisons=east_comparisons,errors=east_errors), per_window=datasets,
        training_recipe_id=seal["recipe_id"], scoring_recipe_id=recipe,
        adapter_hashes={v: [sha256_file(p) for p in ps] for v, ps in paths.items()},
        matched_epoch_traces=True, full_fit_exposure=True,
        non_claims=["All444windows consumed development; no new independent confirmation or promotion.",
            "PANGAEA background is published coverage, not reviewed absence of fire; spatial groups are not dated fires.",
            "No field truth, severity/L2A useful gain, global ranking or original goal completion."])
    guard_output(args.runs,1024*1024)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print("RESULT", json.dumps({k: v for k, v in result.items() if k in ("eligible", "guards")}), flush=True)


if __name__ == "__main__":
    main()
