"""SPEC-94: fixed-boundary balanced severity versus equal weights and dNBR."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import bind_cache, experiment_fingerprint, sha256_file, verify_cache
from src.comp.hls import tta8
from src.comp.paired_severity import (
    PairedSeverityUNet, apply_fixed_boundary, grouped_severity_comparison,
    mtbs_evaluation_labels, severity_confusion, severity_metrics,
)
from src.comp.terrain import water_ndwi
from src.comp.severity_layers import evaluation_fmask
from scripts.forest_pair_train import load_frozen, read_sample
from scripts.score_forest_pair import infer as infer_boundary


def severity_acceptance(groups, candidate, equal, hybrid):
    versus_hybrid = grouped_severity_comparison(groups, candidate, hybrid)
    versus_equal = grouped_severity_comparison(groups, candidate, equal)
    class_delta = np.asarray(versus_hybrid["candidate"]["per_class"]) - versus_hybrid["baseline"]["per_class"]
    accepted = (versus_hybrid["delta"] >= .02 and versus_hybrid["ci95"][0] > 0
                and versus_equal["delta"] >= 0 and np.all(class_delta >= -.02))
    return dict(versus_hybrid=versus_hybrid, versus_equal=versus_equal,
                per_class_delta=class_delta.tolist(), accepted=bool(accepted))


def predict_severity(root, records, checkpoints, variant, manifest_hash, cache):
    files = [root / r["files"][k] for r in records for k in sorted(r["files"])]
    sources = [Path(__file__), Path("src/comp/paired_severity.py"), Path("src/comp/forest_pair.py"),
               Path("src/comp/hls.py"), Path("scripts/forest_pair_train.py")]
    signature = experiment_fingerprint(records, files, checkpoints, sources,
                                       dict(variant=variant, tta=8, precision="bf16"))
    shape = (len(records), 3, records[0]["grid"]["height"], records[0]["grid"]["width"])
    if cache.exists():
        verify_cache(cache, signature); probabilities = np.load(cache)
        if probabilities.shape != shape: raise ValueError("severity cache shape differs")
        return probabilities
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probabilities = np.zeros(shape, np.float32)
    for seed, path in enumerate(checkpoints, 1):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        summary = json.loads(path.with_name("summary.json").read_text())
        if (bundle["seed"] != seed or bundle["variant"] != variant or bundle["epochs"] != 120
                or bundle["width"] != 32 or bundle["depth"] != 7 or bundle["classes"] != 3
                or bundle["smoke"] or bundle["data_manifest"] != manifest_hash
                or not summary["completed"] or summary["fingerprint"] != bundle["fingerprint"]):
            raise ValueError("severity checkpoint violates frozen recipe")
        net = PairedSeverityUNet().to(device).eval(); net.load_state_dict(bundle["state"])
        mean, std = np.asarray(bundle["mean"]), np.asarray(bundle["std"])
        for i, record in enumerate(records):
            image, _, _ = read_sample(root, record, "pair")
            tensor = torch.from_numpy(((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(device)
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                prob = tta8(net, tensor).softmax(1)[0].float().cpu().numpy()
            if not np.isfinite(prob).all(): raise FloatingPointError("nonfinite severity prediction")
            probabilities[i] += prob / len(checkpoints)
        del net
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("xb") as stream: np.save(stream, probabilities)
    bind_cache(cache, signature)
    return probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): raise ValueError("immutable result already exists")
    records = [r for r in load_frozen(args.data_root, args.manifest) if r["partition"] == "evaluation"]
    manifest_hash = sha256_file(args.manifest)
    checkpoints = {v: [args.runs / f"{v}-s{s}" / "model.pt" for s in (1, 2, 3)]
                   for v in ("balanced", "equal")}
    probabilities = {v: predict_severity(args.data_root, records, checkpoints[v], v, manifest_hash,
                                        args.out.parent / (args.out.stem + "-" + v + ".npy")) for v in checkpoints}
    references = [args.reference_root / f"hls-c1mm-final-s{s}" / "model.pt" for s in (1, 2, 3, 4, 5)]
    boundaries = infer_boundary(args.data_root, records, references, "c1mm", manifest_hash,
                                args.out.parent / (args.out.stem + "-boundary.npy"))
    counts = {v: [] for v in ("balanced", "equal", "hybrid")}; conditional = {v: [] for v in counts}
    rows = []
    for i, record in enumerate(records):
        image, _, valid = read_sample(args.data_root, record, "pair")
        with rasterio.open(args.data_root / record["files"]["severity"]) as source: raw = source.read(1)
        truth = mtbs_evaluation_labels(raw, valid); known = truth >= 0
        fmask, fmask_hash = evaluation_fmask(args.data_root, record)
        post = image[6:]; water = water_ndwi(fmask, post[1], post[3])
        boundary = (boundaries[i] >= .5) & ~water
        nbr = lambda x: (x[3] - x[5]) / (x[3] + x[5] + 1e-6)
        dnbr = nbr(image[:6]) - nbr(post); levels = np.digitize(dnbr, [.27, .66]) + 1
        predictions = {v: apply_fixed_boundary(probabilities[v][i], boundary, known) for v in probabilities}
        predictions["hybrid"] = np.where(boundary & known, levels, 0)
        row = dict(sample_id=record["sample_id"], group_id=record["group_id"],
                   valid_pixels=int(known.sum()), fmask_sha256=fmask_hash)
        for v, pred in predictions.items():
            cm = severity_confusion(truth, pred); counts[v].append(cm); row[v] = cm.tolist()
            grade_pred = levels if v == "hybrid" else probabilities[v][i].argmax(0) + 1
            conditional_truth = np.where(truth > 0, truth, -1)
            conditional[v].append(severity_confusion(conditional_truth, grade_pred))
        rows.append(row)
    result = severity_acceptance([r["group_id"] for r in records], counts["balanced"], counts["equal"], counts["hybrid"])
    result.update(spec_id="SPEC-94", data_manifest=manifest_hash, per_window=rows,
                  on_true_burn={v: severity_metrics(np.sum(cm, axis=0)) for v, cm in conditional.items()},
                  model_hashes={v: [sha256_file(p) for p in paths] for v, paths in checkpoints.items()},
                  boundary_hashes=[sha256_file(p) for p in references],
                  non_claims=["MTBS satellite interpretation, not field truth.",
                              "Development evaluation after adaptive experimentation.",
                              "Not validated severity in Siberia or Cyprus."])
    with args.out.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}), flush=True)


if __name__ == "__main__": main()
