"""SPEC-110: CPU replay of legacy severity confusion decomposed by raw MTBS class."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import score_paired_severity, score_forest_pair
from scripts.forest_pair_train import read_sample
from scripts.prepare_fresh_paired_severity import read_bound_fmask
from src.comp.accuracy_artifacts import execution_sources
from src.comp.accuracy_contract import canonical_hash, experiment_fingerprint, sha256_file, verify_cache
from src.comp.mtbs_legend_audit import attribute
from src.comp.paired_severity import apply_fixed_boundary, mtbs_evaluation_labels
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi

POPULATIONS = (
    ("paired-severity-v1-result", "severity-pairs/manifest.json", 172,
     "5f6b2839b89c807692dc4c931b2f4e713ae03294f869cd5233faf89a5617999e",
     "5f80002e316fb0f7a0170ff16b3cf19ade07b5fcea5649885ad09247739edfd6"),
    ("fresh-covered-v1-result", "severity-covered-v1/manifest.json", 173,
     "c8721c6a1938912c514afea4008526e248681c547b9496e715f6140eb9d0fa62",
     "ed7903750216a4e341a394908dbe06f44fd195e0ade8f67034ec8497a83e79dc"),
)


def cached(root, records, paths, kind, cache):
    if not cache.exists() or not cache.with_name(cache.name+".binding.json").exists():
        raise ValueError("completed cache missing; no inference allowed")
    inputs = [root/r["files"][k] for r in records for k in sorted(r["files"])]
    sources = execution_sources(Path(score_forest_pair.__file__), "scripts/forest_pair_train.py") if kind == "c1mm" else execution_sources(
        Path(score_paired_severity.__file__), "src/comp/paired_severity.py", "scripts/forest_pair_train.py")
    signature = experiment_fingerprint(records, inputs, paths, sources, dict(variant=kind, tta=8, precision="bf16"))
    verify_cache(cache, signature)
    array = np.load(cache, allow_pickle=False, mmap_mode="r")
    shape = (len(records), 512, 512) if kind == "c1mm" else (len(records), 3, 512, 512)
    if array.shape != shape:
        raise ValueError("completed cache shape differs")
    return array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("immutable legend audit exists")
    reports = {}
    for stem, relative, count, parent_sha, manifest_sha in POPULATIONS:
        parent_file = Path("research")/(stem+".json")
        manifest = args.data_root/relative
        if sha256_file(parent_file) != parent_sha or sha256_file(manifest) != manifest_sha:
            raise ValueError("frozen severity parent/population differs")
        parent = json.loads(parent_file.read_text())
        records = [r for r in json.loads(manifest.read_text())["records"] if r["partition"] == "evaluation"]
        if len(records) != count or len(parent["per_window"]) != count:
            raise ValueError("severity population incomplete")
        for record in records:
            for key, relative_file in record["files"].items():
                if sha256_file(args.data_root/relative_file) != record["hashes"][key]:
                    raise ValueError("frozen severity input changed")
        variants = ["equal", "balanced"] if stem.startswith("paired") else ["equal"]
        probabilities = {}
        for variant in variants:
            paths = [Path("research/paired-severity-v1")/f"{variant}-s{s}/model.pt" for s in (1, 2, 3)]
            expected = parent["model_hashes"][variant] if stem.startswith("paired") else parent["model_hashes"]
            if [sha256_file(path) for path in paths] != expected:
                raise ValueError("severity model identities changed")
            probabilities[variant] = cached(args.data_root, records, paths, variant, Path("research")/(stem+"-"+variant+".npy"))
        paths = [args.reference_root/f"hls-c1mm-final-s{s}/model.pt" for s in (1, 2, 3, 4, 5)]
        if [sha256_file(path) for path in paths] != parent["boundary_hashes"]:
            raise ValueError("contour reference changed")
        boundaries = cached(args.data_root, records, paths, "c1mm", Path("research")/(stem+"-boundary.npy"))
        rows = []
        for i, (record, old_row) in enumerate(zip(records, parent["per_window"])):
            if record["sample_id"] != old_row["sample_id"] or record["group_id"] != old_row["group_id"]:
                raise ValueError("severity identities/order differ")
            image, _, valid = read_sample(args.data_root, record, "pair")
            with rasterio.open(args.data_root/record["files"]["severity"]) as source:
                raw = source.read(1)
            if "fmask" in record["files"]:
                fm, fm_sha = read_bound_fmask(args.data_root, record)
            else:
                legacy = args.data_root/record["files"]["post"].replace("_merged.tif", ".fmask.tif")
                own = args.data_root/"severity-fmask"/(canonical_hash(dict(item=record["post_item"], grid=record["grid"]))+".tif")
                if not legacy.exists() and not (own.exists() and own.with_suffix(".json").exists()):
                    raise ValueError("cached Fmask absent; no download allowed")
                fm, fm_sha = evaluation_fmask(args.data_root, record)
            if fm_sha != old_row["fmask_sha256"]:
                raise ValueError("legacy observation mask changed")
            known = mtbs_evaluation_labels(raw, valid) >= 0
            boundary = (boundaries[i] >= .5) & ~water_ndwi(fm, image[7], image[9])
            nbr = lambda x: (x[3]-x[5])/(x[3]+x[5]+1e-6)
            levels = np.digitize(nbr(image[:6])-nbr(image[6:]), [.27, .66])+1
            predictions = {v: apply_fixed_boundary(a[i], boundary, known) for v, a in probabilities.items()}
            predictions["hybrid"] = np.where(boundary & known, levels, 0)
            audit = {v: attribute(raw, predicted, valid) for v, predicted in predictions.items()}
            for variant, values in audit.items():
                if values["legacy_confusion"] != old_row[variant]:
                    raise ValueError("archived confusion does not reconstruct")
            rows.append(dict(sample_id=record["sample_id"], group_id=record["group_id"], audit=audit))
        reports[stem] = dict(parent_sha256=parent_sha, manifest_sha256=manifest_sha, windows=count, rows=rows)
        print("ATTRIBUTED", stem, count, flush=True)
    result = dict(spec_id="SPEC-110", populations=reports, all_legacy_confusions_exact=True,
        source_hashes={p: sha256_file(p) for p in (str(Path(__file__)), "src/comp/mtbs_legend_audit.py")},
        non_claims=["Legacy benchmark conventions preserved; no easier replacement metric or model promotion.",
            "MTBS0 is outside assessment and5 is increased greenness, not independently reviewed physical fire absence.",
            "No new accuracy, causal error proof, field severity, changed contours/labels or goal completion."])
    content = json.dumps(result, separators=(",", ":"), allow_nan=False)
    if len(content.encode()) > 2*1024**2:
        raise ValueError("compact audit limit exceeded")
    with args.out.open("x") as stream:
        stream.write(content+"\n")


if __name__ == "__main__":
    main()
