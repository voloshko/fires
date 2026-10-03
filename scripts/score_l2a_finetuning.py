"""SPEC-93: source adaptation on identical L2A pixels with two frozen controls."""
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
    bind_cache, experiment_fingerprint, paired_group_comparison, sha256_file, verify_cache,
)
from src.comp.forest_pair import make_forest_model
from src.comp.hls import tta8
from src.comp.l2a_finetuning import l2a_water
from scripts.l2a_finetune import load_twins, read_twin
from scripts.score_forest_pair import binary_counts
from src.comp.accuracy_artifacts import assert_matched_recipes, execution_sources, load_checkpoint


def l2a_acceptance(records, candidate, hls, initial):
    groups = [r["group_id"] for r in records]
    overall = {"continued_hls": paired_group_comparison(groups, candidate, hls),
               "initial_c1mm": paired_group_comparison(groups, candidate, initial)}
    by_dataset = {}
    for dataset in ("ems", "floga"):
        idx = [i for i, r in enumerate(records) if r["dataset"] == dataset]
        if not idx: raise ValueError("required L2A evaluation dataset is missing")
        by_dataset[dataset] = {name: paired_group_comparison(
            [groups[i] for i in idx], [candidate[i] for i in idx], [baseline[i] for i in idx])
            for name, baseline in (("continued_hls", hls), ("initial_c1mm", initial))}
    accepted = all(r["delta"] >= .01 and r["ci95"][0] > 0 for r in overall.values())
    accepted &= all(r["delta"] >= -.01 and r["ci95"][0] >= -.03
                    for ds in by_dataset.values() for r in ds.values())
    return dict(overall=overall, by_dataset=by_dataset, accepted=bool(accepted))


def predict(root, records, models, variant, manifest_hash, cache):
    inputs = [root / r["files"][k] for r in records for k in sorted(r["files"])]
    sources = execution_sources(Path(__file__), "scripts/l2a_finetune.py", "src/comp/l2a_finetuning.py")
    signature = experiment_fingerprint(records, inputs, models, sources,
                                       dict(model_variant=variant, input_source="l2a", tta=8, precision="bf16"))
    shape = (len(records), records[0]["grid"]["height"], records[0]["grid"]["width"])
    if cache.exists():
        verify_cache(cache, signature); probabilities = np.load(cache)
        if probabilities.shape != shape: raise ValueError("L2A prediction shape differs")
        return probabilities
    dev = "cuda" if torch.cuda.is_available() else "cpu"; probabilities = np.zeros(shape, np.float32)
    for seed, path in enumerate(models, 1):
        bundle = load_checkpoint(path, sha256_file(path), export_root=Path("research/tensor-exports"))
        if variant != "initial":
            summary = json.loads(path.with_name("summary.json").read_text())
            if (bundle["seed"] != seed or bundle["variant"] != variant or bundle["epochs"] != 30
                    or bundle["w"] != 32 or bundle["depth"] != 7 or bundle["smoke"]
                    or bundle["data_manifest"] != manifest_hash or not summary["completed"]
                    or summary["fingerprint"] != bundle["fingerprint"]):
                raise ValueError("finetuning checkpoint violates frozen recipe")
        net = make_forest_model("post", bundle["w"], 7).to(dev).eval(); net.load_state_dict(bundle["state"])
        mean, std = np.asarray(bundle["mean"]), np.asarray(bundle["std"])
        for i, record in enumerate(records):
            image, _, _ = read_twin(root, record, "l2a")
            tensor = torch.from_numpy(((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                prob = tta8(net, tensor).softmax(1)[0, 1].float().cpu().numpy()
            if not np.isfinite(prob).all(): raise FloatingPointError("nonfinite L2A prediction")
            probabilities[i] += prob / len(models)
        del net
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("xb") as stream: np.save(stream, probabilities)
    bind_cache(cache, signature)
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists(): raise ValueError("immutable result already exists")
    records = [r for r in load_twins(args.data_root, args.manifest) if r["partition"] == "evaluation"]
    manifest_hash = sha256_file(args.manifest)
    models = {v: [args.runs / f"{v}-s{s}" / "model.pt" for s in (1, 2, 3)] for v in ("hls", "l2a")}
    models["initial"] = [args.reference_root / f"hls-c1mm-final-s{s}" / "model.pt" for s in (1, 2, 3, 4, 5)]
    archived = [json.loads(path.with_name("data_manifest.json").read_text())
                for v in ("hls", "l2a") for path in models[v]]
    recipe_id = assert_matched_recipes(archived, ("w", "depth", "epochs", "classes", "initialization_family"))
    for seed in (1, 2, 3):
        expected = sha256_file(args.reference_root / f"hls-c1mm-final-s{seed}" / "model.pt")
        for variant in ("hls", "l2a"):
            record = json.loads((args.runs / f"{variant}-s{seed}" / "data_manifest.json").read_text())
            if record["initialization_sha256"] != expected:
                raise ValueError("continuation does not use its corresponding C1-MM initializer")
    probabilities = {v: predict(args.data_root, records, paths, v, manifest_hash,
                                args.out.parent / (args.out.stem + "-" + v + ".npy")) for v, paths in models.items()}
    counts = {v: [] for v in models}; rows = []
    for i, record in enumerate(records):
        image, truth, valid = read_twin(args.data_root, record, "l2a")
        with rasterio.open(args.data_root / record["files"]["scl"]) as source: scl = source.read(1)
        water = l2a_water(image, scl)
        row = dict(sample_id=record["sample_id"], group_id=record["group_id"], dataset=record["dataset"],
                   valid_pixels=int(valid.sum()))
        for v, prob in probabilities.items():
            cm = binary_counts(truth == 1, (prob[i] >= .5) & ~water, valid)
            counts[v].append(cm); row[v] = cm
        rows.append(row)
    result = l2a_acceptance(records, counts["l2a"], counts["hls"], counts["initial"])
    result.update(spec_id="SPEC-93", recipe_id=recipe_id, data_manifest=manifest_hash, per_window=rows,
                  models={v: [{"path": str(p), "sha256": sha256_file(p)} for p in paths] for v, paths in models.items()},
                  non_claims=["Development evaluation after adaptive experimentation.",
                              "No independent Siberian transfer or field-truth claim.",
                              "All models evaluated on the same L2A input pixels."])
    with args.out.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}), flush=True)


if __name__ == "__main__": main()
