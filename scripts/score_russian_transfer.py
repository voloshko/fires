"""SPEC-92: same-pixel reserve/control/C1-MM/Prithvi comparison on untouched Russia."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import bind_cache, canonical_hash, experiment_fingerprint, paired_group_comparison, sha256_file, verify_cache
from src.comp.accuracy_artifacts import checkpoint_metadata, execution_sources, load_checkpoint
from src.comp.forest_pair import make_forest_model
from src.comp.hls import tta8
from src.comp.l2a_finetuning import l2a_water
from src.comp.russian_transfer import transfer_acceptance
from src.comp.severity_layers import evaluation_fmask
from src.comp.terrain import water_ndwi
from scripts.forest_pair_train import load_frozen, read_sample
from scripts.russian_transfer_train import load_russian
from scripts.score_forest_pair import binary_counts


def model_recipe_audit(models, base_hash, russian_hash, initializers, base_fit, russian_fit):
    metadata = {v: [checkpoint_metadata(p, export_root=Path("research/tensor-exports")) for p in paths]
                for v, paths in models.items()}
    values = [r for rows in metadata.values() for r in rows]
    fields = ("source_hashes", "software", "base_manifest", "russian_manifest", "width", "depth",
              "epochs", "batch", "steps_per_epoch", "classes")
    for field in fields:
        if any(v.get(field) != values[0].get(field) for v in values): raise ValueError("transfer recipe drift: " + field)
    for variant, rows in metadata.items():
        for seed, row in enumerate(rows, 1):
            expected = set(base_fit) | (set(russian_fit) if variant == "reserve" else set())
            if (row["seed"] != seed or row["variant"] != variant or row["smoke"] or row["epochs"] != 30
                    or row["base_manifest"] != base_hash or row["russian_manifest"] != russian_hash
                    or set(row["fit_ids"]) != expected or row["initialization_sha256"] != sha256_file(initializers[seed - 1])):
                raise ValueError("transfer checkpoint recipe/lineage mismatch")
    return canonical_hash({k: values[0][k] for k in fields})


def predict(root, records, paths, kind, cache, prithvi_root):
    files = [root / r["files"][k] for r in records for k in sorted(r["files"])]
    sources = execution_sources(Path(__file__), "scripts/russian_transfer_train.py", "src/comp/russian_transfer.py",
                                "scripts/forest_pair_train.py")
    if kind == "prithvi": paths = [prithvi_root / "Prithvi_EO_V2_300M_BurnScars.pt", prithvi_root / "burn_scars_config.yaml"]
    signature = experiment_fingerprint(records, files, paths, sources, dict(kind=kind, tta=8, precision="bf16"))
    shape = (len(records), records[0]["grid"]["height"], records[0]["grid"]["width"])
    if cache.exists():
        verify_cache(cache, signature); p = np.load(cache)
        if p.shape != shape: raise ValueError("transfer prediction cache shape mismatch")
        return p
    dev = "cuda" if torch.cuda.is_available() else "cpu"; p = np.zeros(shape, np.float32)
    if kind == "prithvi":
        from terratorch.cli_tools import LightningInferenceModel
        cfg = yaml.safe_load((prithvi_root / "burn_scars_config.yaml").read_text())
        mean = np.array(cfg["data"]["init_args"]["means"], np.float32)
        std = np.array(cfg["data"]["init_args"]["stds"], np.float32)
        cfg["model"]["init_args"]["model_args"]["backbone_pretrained"] = False
        cfg["trainer"]["logger"] = False; cfg["trainer"]["callbacks"] = []
        cfg["trainer"]["enable_checkpointing"] = False
        inference_cfg = cache.with_suffix(".prithvi-config.yaml")
        if not inference_cfg.exists(): inference_cfg.write_text(yaml.safe_dump(cfg))
        elif yaml.safe_load(inference_cfg.read_text()) != cfg: raise ValueError("Prithvi inference config changed")
        model = LightningInferenceModel.from_config(inference_cfg, paths[0]).model.to(dev).eval()
        wrappers = [(model, mean, std)]
    else:
        wrappers = []
        for path in paths:
            b = load_checkpoint(path, sha256_file(path), export_root=Path("research/tensor-exports"))
            pair = kind in ("control", "reserve")
            net = make_forest_model("pair" if pair else "post", b.get("width", b.get("w", 32)), 7)
            net.load_state_dict(b["state"]); wrappers.append((net, np.asarray(b["mean"]), np.asarray(b["std"])))
    for net, mean, std in wrappers:
        net = net.to(dev).eval()
        forward = (lambda x: (lambda y: getattr(y, "output", y))(net(x))) if kind == "prithvi" else net
        for i, record in enumerate(records):
            image, _, _ = read_sample(root, record, "pair" if kind in ("control", "reserve") else "post")
            x = torch.from_numpy(((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].to(dev)
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                probability = tta8(forward, x).softmax(1)[0, 1].float().cpu().numpy()
            if not np.isfinite(probability).all(): raise FloatingPointError("nonfinite transfer prediction")
            p[i] += probability / len(wrappers)
        net.cpu()
    del wrappers
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("xb") as stream: np.save(stream, p)
    bind_cache(cache, signature)
    return p


def counts_for(root, records, predictions):
    counts = {v: [] for v in predictions}; rows = []
    for i, record in enumerate(records):
        image, labels, valid = read_sample(root, record, "post")
        if record.get("date_semantics") == "annual-interval-pair":
            with rasterio.open(root / record["files"]["post_scl"]) as source: scl = source.read(1)
            water = l2a_water(image, scl)
        else:
            fmask, _ = evaluation_fmask(root, record); water = water_ndwi(fmask, image[1], image[3])
        row = dict(sample_id=record["sample_id"], group_id=record["group_id"], valid_pixels=int(valid.sum()))
        for name, p in predictions.items():
            cm = binary_counts(labels == 1, (p[i] >= .5) & ~water, valid)
            counts[name].append(cm); row[name] = cm
        rows.append(row)
    return counts, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "base-manifest", "russian-manifest", "runs", "initializer-root", "reference-root", "prithvi-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists(): raise ValueError("immutable transfer result already exists")
    base, russian = load_frozen(a.data_root, a.base_manifest), load_russian(a.data_root, a.russian_manifest)
    evaluation = [r for r in russian if r["partition"] == "evaluation"]
    models = {v: [a.runs / f"{v}-s{s}" / "model.pt" for s in (1, 2, 3)] for v in ("control", "reserve")}
    initializers = [a.initializer_root / f"pair-s{s}" / "model.pt" for s in (1, 2, 3)]
    recipe = model_recipe_audit(models, sha256_file(a.base_manifest), sha256_file(a.russian_manifest), initializers,
                              [r["sample_id"] for r in base if r["partition"] == "fit"],
                              [r["sample_id"] for r in russian if r["partition"] == "fit"])
    references = [a.reference_root / f"hls-c1mm-final-s{s}" / "model.pt" for s in (1, 2, 3, 4, 5)]
    all_models = dict(models, c1mm=references, prithvi=[])
    predictions = {v: predict(a.data_root, evaluation, paths, v, a.out.parent / (a.out.stem + "-" + v + ".npy"), a.prithvi_root)
                   for v, paths in all_models.items()}
    counts, rows = counts_for(a.data_root, evaluation, predictions); groups = [r["group_id"] for r in evaluation]
    comparisons = {v: paired_group_comparison(groups, counts["reserve"], counts[v]) for v in ("control", "c1mm", "prithvi")}
    stability = {}
    for dataset in ("ems", "floga"):
        records = [r for r in base if r["partition"] == "evaluation" and r["dataset"] == dataset]
        preds = {v: predict(a.data_root, records, paths, v, a.out.parent / (a.out.stem + "-" + dataset + "-" + v + ".npy"), a.prithvi_root)
                 for v, paths in dict(models, c1mm=references).items()}
        cm, _ = counts_for(a.data_root, records, preds)
        stability[dataset] = paired_group_comparison([r["group_id"] for r in records], cm["reserve"], cm["c1mm"])
    decision = transfer_acceptance(comparisons["control"], stability, comparisons["prithvi"])
    result = dict(spec_id="SPEC-92", recipe_id=recipe, comparisons=comparisons, stability=stability, **decision,
                  per_window=rows, russian_manifest=sha256_file(a.russian_manifest), base_manifest=sha256_file(a.base_manifest),
                  model_hashes={v: [sha256_file(p) for p in paths] for v, paths in dict(models, c1mm=references).items()},
                  competitor_weights_sha256=sha256_file(a.prithvi_root / "Prithvi_EO_V2_300M_BurnScars.pt"),
                  non_claims=["Spring2020 participatory interval labels, not field truth.",
                              "No claim of superiority on all regions or different benchmark datasets."])
    with a.out.open("x") as stream: json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "per_window"}), flush=True)


if __name__ == "__main__": main()
