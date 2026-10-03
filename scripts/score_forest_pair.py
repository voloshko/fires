"""SPEC-91: compare frozen three-seed pair/post ensembles and C1-MM."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import (
    bind_cache, experiment_fingerprint, paired_group_comparison, pooled_iou,
    sha256_file, verify_cache,
)
from src.comp.forest_pair import make_forest_model
from src.comp.hls import tta8
from src.comp.terrain import water_ndwi
from scripts.forest_pair_train import load_frozen, read_sample
from src.comp.accuracy_artifacts import assert_matched_recipes, execution_sources, load_checkpoint


def binary_counts(truth, prediction, valid):
    truth, prediction, valid = np.asarray(truth, bool), np.asarray(prediction, bool), np.asarray(valid, bool)
    if truth.shape != prediction.shape or truth.shape != valid.shape: raise ValueError("unaligned pixels")
    return [int((truth & prediction & valid).sum()), int((~truth & prediction & valid).sum()),
            int((truth & ~prediction & valid).sum())]


def acceptance_report(records, candidate, control, reference):
    groups = [r["group_id"] for r in records]
    comparisons = {"matched_post": paired_group_comparison(groups, candidate, control),
                   "c1mm": paired_group_comparison(groups, candidate, reference)}
    per_dataset = {}
    for dataset in ("ems", "floga"):
        idx = [i for i, r in enumerate(records) if r["dataset"] == dataset]
        if not idx: raise ValueError("missing required evaluation dataset")
        per_dataset[dataset] = {}
        for name, baseline in (("matched_post", control), ("c1mm", reference)):
            per_dataset[dataset][name] = paired_group_comparison(
                [groups[i] for i in idx], [candidate[i] for i in idx], [baseline[i] for i in idx])
    def lost_events(counts):
        sums = {}
        for group, count in zip(groups, counts):
            old = sums.setdefault(group, [0, 0])
            old[0] += count[0]; old[1] += count[0] + count[2]
        return sum(tp == 0 and true_area > 0 for tp, true_area in sums.values())
    lost = {name: lost_events(values) for name, values in
            (("pair", candidate), ("post", control), ("c1mm", reference))}
    accepted = all(r["delta"] >= .01 and r["ci95"][0] > 0 for r in comparisons.values())
    accepted &= all(r["delta"] >= -.01 and r["ci95"][0] >= -.03
                    for values in per_dataset.values() for r in values.values())
    accepted &= lost["pair"] <= lost["c1mm"]
    return {"comparisons": comparisons, "per_dataset": per_dataset,
            "lost_events": lost, "accepted": bool(accepted),
            "pooled_iou": {name: pooled_iou(values) for name, values in
                           (("pair", candidate), ("post", control), ("c1mm", reference))}}


def infer(root, records, models, variant, manifest_hash, cache):
    files = [root / r["files"][key] for r in records for key in sorted(r["files"])]
    sources = execution_sources(Path(__file__), "scripts/forest_pair_train.py")
    signature = experiment_fingerprint(records, files, models, sources,
                                       {"variant": variant, "tta": 8, "precision": "bf16"})
    if cache.exists():
        verify_cache(cache, signature)
        probabilities = np.load(cache)
        if probabilities.shape != (len(records), records[0]["grid"]["height"], records[0]["grid"]["width"]):
            raise ValueError("prediction cache shape mismatch")
        return probabilities
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probabilities = np.zeros((len(records), records[0]["grid"]["height"], records[0]["grid"]["width"]), np.float32)
    for seed, path in enumerate(models, 1):
        bundle = load_checkpoint(path, sha256_file(path), export_root=Path("research/tensor-exports"))
        if variant != "c1mm":
            summary = json.loads(path.with_name("summary.json").read_text())
            if (bundle["variant"] != variant or bundle["seed"] != seed or bundle["smoke"]
                    or bundle["epochs"] != 120 or bundle["width"] != 32 or bundle["depth"] != 7
                    or bundle["data_manifest"] != manifest_hash or not summary["completed"]
                    or summary["fingerprint"] != bundle["fingerprint"]):
                raise ValueError("model does not match the frozen recipe")
            net = make_forest_model(variant, bundle["width"], bundle["depth"])
        else:
            net = make_forest_model("post", bundle["w"], 7)
        net.load_state_dict(bundle["state"]); net.to(dev).eval()
        mean, std = np.asarray(bundle["mean"]), np.asarray(bundle["std"])
        for i, record in enumerate(records):
            if variant == "c1mm":
                with rasterio.open(root / record["files"]["post"]) as source: x = source.read().astype(np.float32)
                with rasterio.open(root / record["files"]["mask"]) as source: y = source.read(1)
                valid = (x != -9999).all(0) & np.isfinite(x).all(0) & (y >= 0)
                x = np.where(valid[None], x, 0)
            else: x, _, _ = read_sample(root, record, variant)
            tensor = torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                prediction = tta8(net, tensor).softmax(1)[0, 1].float().cpu().numpy()
            if not np.isfinite(prediction).all(): raise FloatingPointError("nonfinite inference")
            probabilities[i] += prediction / len(models)
        del net
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("xb") as stream: np.save(stream, probabilities)
    bind_cache(cache, signature)
    return probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--runs", required=True, type=Path)
    p.add_argument("--reference-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists(): raise ValueError("result already exists")
    records = [r for r in load_frozen(args.data_root, args.manifest) if r["partition"] == "evaluation"]
    checkpoints = {variant: [args.runs / f"{variant}-s{s}" / "model.pt" for s in (1, 2, 3)]
                   for variant in ("post", "pair")}
    checkpoints["c1mm"] = [args.reference_root / f"hls-c1mm-final-s{s}" / "model.pt" for s in (1, 2, 3, 4, 5)]
    archived = [json.loads(path.with_name("data_manifest.json").read_text())
                for key in ("post", "pair") for path in checkpoints[key]]
    recipe_id = assert_matched_recipes(archived, ("width", "depth", "epochs", "batch", "evaluation_ids"))
    predictions = {key: infer(args.data_root, records, models, key, sha256_file(args.manifest),
                              args.out.parent / (args.out.stem + "-" + key + ".npy"))
                   for key, models in checkpoints.items()}
    counts = {key: [] for key in predictions}; per_window = []
    for i, record in enumerate(records):
        post, labels, valid = read_sample(args.data_root, record, "post")
        fmask_path = args.data_root / record["files"]["post"].replace("_merged.tif", ".fmask.tif")
        with rasterio.open(fmask_path) as source:
            if source.crs != rasterio.crs.CRS.from_string(record["grid"]["crs"]) or list(source.transform)[:6] != record["grid"]["transform"]:
                raise ValueError("water Fmask grid differs")
            fmask = source.read(1)
        water = water_ndwi(fmask, post[1], post[3])
        row = {"sample_id": record["sample_id"], "group_id": record["group_id"],
               "dataset": record["dataset"], "valid_pixels": int(valid.sum()),
               "fmask_sha256": sha256_file(fmask_path)}
        for key, prob in predictions.items():
            cm = binary_counts(labels == 1, (prob[i] >= .5) & ~water, valid)
            counts[key].append(cm); row[key] = cm
        per_window.append(row)
    result = acceptance_report(records, counts["pair"], counts["post"], counts["c1mm"])
    result.update(spec_id="SPEC-91", recipe_id=recipe_id, data_manifest=sha256_file(args.manifest),
                  models={key: [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
                          for key, paths in checkpoints.items()}, per_window=per_window,
                  non_claims=["Development revalidation after adaptive model selection.",
                              "Not an independent hidden test or field truth.",
                              "No global ranking from metrics on different datasets."])
    with args.out.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}), flush=True)


if __name__ == "__main__": main()
