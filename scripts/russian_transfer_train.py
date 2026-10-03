"""SPEC-92: equal-step continuation with/without mined Russian training reserve."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rasterio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import experiment_fingerprint, sha256_file, validate_manifest
from src.comp.accuracy_artifacts import execution_sources, load_checkpoint, save_checkpoint
from src.comp.forest_pair import augment_forest, forest_loss, make_forest_model
from src.comp.hls import tta8
from src.comp.l2a_finetuning import l2a_water
from src.comp.russian_transfer import hard_background_weights, validate_pair_initializer
from scripts.forest_pair_train import load_frozen, normalization, read_sample
from scripts.prepare_russian_windows import interval_pair_valid


def load_russian(root, manifest):
    payload = json.loads(manifest.read_text())
    if not payload["ready_for_training"] or payload["qc_probe"]:
        raise ValueError("Russian data gate did not authorize training")
    records = payload["records"]; validate_manifest(records)
    for record in records:
        interval_pair_valid(record)
        for key, path in record["files"].items():
            if sha256_file(root / path) != record["hashes"][key]: raise ValueError("Russian source input changed")
    return records


def mine(root, records, net, mean, std):
    if any(r["partition"] != "fit" for r in records): raise ValueError("mining saw evaluation or calibration")
    counts = {}; net.eval()
    for record in records:
        image, labels, valid = read_sample(root, record, "pair")
        x = torch.from_numpy(((image - mean[:, None, None]) / std[:, None, None]).astype(np.float32))[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            probability = tta8(net, x).softmax(1)[0, 1].float().cpu().numpy()
        with rasterio.open(root / record["files"]["post_scl"]) as source: scl = source.read(1)
        predicted = (probability >= .5) & ~l2a_water(image[6:], scl)
        counts[record["sample_id"]] = int((predicted & (labels == 0) & valid).sum())
    return counts, hard_background_weights(records, counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--russian-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initializer-result", type=Path, default=Path("research/forest-pair-v1-result.json"))
    p.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    p.add_argument("--variant", choices=("control", "reserve"), required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.out.exists(): raise ValueError("immutable run already exists")
    base = load_frozen(args.data_root, args.base_manifest)
    russian = load_russian(args.data_root, args.russian_manifest)
    validate_manifest(base + russian)
    base_fit = [r for r in base if r["partition"] == "fit"]
    russian_fit = [r for r in russian if r["partition"] == "fit"]
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    bundle = load_checkpoint(args.checkpoint, sha256_file(args.checkpoint), export_root=Path("research/tensor-exports"))
    frozen_result = json.loads(args.initializer_result.read_text())
    if sha256_file(args.checkpoint) != frozen_result["models"]["pair"][args.seed - 1]["sha256"]:
        raise ValueError("initializer is not the archived SPEC-91 checkpoint")
    initializer_recipe = validate_pair_initializer(bundle, args.seed, sha256_file(args.base_manifest), frozen_result["recipe_id"])
    net = make_forest_model("pair").cuda(); net.load_state_dict(bundle["state"])
    initial_mean, initial_std = np.asarray(bundle["mean"]), np.asarray(bundle["std"])
    del bundle
    counts, mining_weights = ({}, {})
    if args.variant == "reserve": counts, mining_weights = mine(args.data_root, russian_fit, net, initial_mean, initial_std)
    fit = base_fit + (russian_fit if args.variant == "reserve" else [])
    weights = np.array([r["sampling_weight"] if r["sample_id"] not in mining_weights else mining_weights[r["sample_id"]]
                        for r in fit], np.float64)
    mean, std = normalization((read_sample(args.data_root, r, "pair") for r in fit), "pair")
    steps_per_epoch = math.ceil(sum(r["sampling_weight"] for r in base_fit) / 8)
    epochs, batch = 30, 8
    sources = execution_sources(Path(__file__), "src/comp/russian_transfer.py", "scripts/forest_pair_train.py",
                                "scripts/prepare_russian_windows.py")
    inputs = [args.data_root / r["files"][key] for r in base + russian for key in sorted(r["files"])]
    signature = experiment_fingerprint(base + russian, inputs, [args.checkpoint], sources,
                dict(variant=args.variant, seed=args.seed, mean=mean.tolist(), std=std.tolist(),
                     steps_per_epoch=steps_per_epoch, epochs=epochs, batch=batch,
                     sampling_weights=weights.tolist(), mining_counts=counts))
    config = dict(spec_id="SPEC-92", variant=args.variant, seed=args.seed, width=32, depth=7,
                  classes=2, epochs=epochs, batch=batch, steps_per_epoch=steps_per_epoch, smoke=False,
                  mean=mean.tolist(), std=std.tolist(), fingerprint=signature,
                  base_manifest=sha256_file(args.base_manifest), russian_manifest=sha256_file(args.russian_manifest),
                  initialization_sha256=sha256_file(args.checkpoint),
                  initializer_recipe_id=initializer_recipe, initializer_result_sha256=sha256_file(args.initializer_result),
                  fit_ids=[r["sample_id"] for r in fit], mining_counts=counts, mining_weights=mining_weights,
                  source_hashes={str(s): sha256_file(s) for s in sources},
                  software={"torch": str(torch.__version__), "numpy": str(np.__version__)})
    args.out.mkdir(parents=True)
    with (args.out / "data_manifest.json").open("x") as stream: json.dump(config, stream, indent=2)
    images, labels = [], []
    for record in fit:
        x, y, _ = read_sample(args.data_root, record, "pair")
        images.append(torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32)))
        labels.append(torch.from_numpy(y))
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    net.train(); losses = []; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        values = []
        for _ in range(steps_per_epoch):
            idx = rng.choice(len(fit), size=batch, replace=True, p=weights / weights.sum())
            x = torch.stack([images[i] for i in idx]).cuda(); y = torch.stack([labels[i] for i in idx]).cuda()
            x, y = augment_forest(x, y, rng); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(x)
            loss = forest_loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError("nonfinite transfer")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step(); scheduler.step(); values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
        print("epoch", epoch + 1, "loss", losses[-1], "seconds", int(time.monotonic() - start), flush=True)
    save_checkpoint(args.out / "model.pt", net.cpu().state_dict(), config)
    result = dict(config, completed=True, loss=losses, seconds=time.monotonic() - start,
                  peak_vram_bytes=torch.cuda.max_memory_allocated())
    with (args.out / "summary.json").open("x") as stream: json.dump(result, stream, indent=2)
    print("COMPLETE", args.variant, args.seed, flush=True)


if __name__ == "__main__": main()
