"""SPEC-91: matched C1 post/pair training on a frozen prepared manifest."""
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
from src.comp.accuracy_contract import (
    canonical_hash, experiment_fingerprint, sha256_file, validate_manifest,
)
from src.comp.forest_pair import augment_forest, forest_loss, make_forest_model


def read_sample(root, record, variant):
    def read(key):
        with rasterio.open(root / record["files"][key]) as raster: return raster.read()
    post, pre = read("post").astype(np.float32), read("pre").astype(np.float32)
    y = read("mask")[0].astype(np.int64)
    valid = read("valid")[0].astype(bool) & (y >= 0)
    valid &= (post != -9999).all(0) & (pre != -9999).all(0)
    valid &= np.isfinite(post).all(0) & np.isfinite(pre).all(0)
    if not valid.any(): raise ValueError("sample has no valid supervision")
    y[~valid] = -1
    x = np.concatenate((pre, post)) if variant == "pair" else post
    return np.where(valid[None], x, 0).astype(np.float32), y, valid


def normalization(samples, variant):
    count = 0; sums = np.zeros(6, np.float64); squares = sums.copy()
    for x, _, valid in samples:
        dates = (x[:6], x[6:]) if variant == "pair" else (x,)
        for date in dates:
            pixels = date[:, valid].astype(np.float64)
            count += pixels.shape[1]; sums += pixels.sum(1); squares += (pixels ** 2).sum(1)
    if count == 0: raise ValueError("no valid fit pixels")
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 0))
    std = np.maximum(std, 1e-6)
    if variant == "pair": mean, std = np.tile(mean, 2), np.tile(std, 2)
    return mean.astype(np.float32), std.astype(np.float32)


def load_frozen(root, manifest):
    payload = json.loads(manifest.read_text())
    if not payload.get("ready_for_training") or payload.get("qc_probe"):
        raise ValueError("manifest does not authorize full training")
    records = payload["records"]; validate_manifest(records, require_pairs=True)
    for record in records:
        for key, relative in record["files"].items():
            if sha256_file(root / relative) != record["hashes"][key]:
                raise ValueError(f"changed input {record['sample_id']} {key}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("post", "pair"), required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--smoke", action="store_true", help="synthetic GPU execution only")
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("full recipe requires a CUDA GPU")
    if (args.out / "summary.json").exists() or (args.out / "model.pt").exists():
        raise ValueError("run already exists; artifacts are immutable")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    width, depth, epochs, batch = (4, 3, 1, 2) if args.smoke else (32, 7, 120, 8)
    sources = [Path(__file__), Path("src/comp/forest_pair.py"), Path("src/comp/accuracy_contract.py"),
               Path("scripts/train_unet.py")]
    if args.smoke:
        channels = 12 if args.variant == "pair" else 6
        samples = [(rng.normal(size=(channels, 32, 32)).astype(np.float32),
                    rng.integers(0, 2, (32, 32)), np.ones((32, 32), bool)) for _ in range(4)]
        fit, records = [], []
        mean, std = normalization(samples, args.variant)
        signature = canonical_hash({"smoke": True, "seed": args.seed, "variant": args.variant,
                                    "sources": [sha256_file(p) for p in sources]})
        weights = np.ones(len(samples))
    else:
        records = load_frozen(args.data_root, args.manifest)
        fit = [r for r in records if r["partition"] == "fit"]
        samples = [read_sample(args.data_root, r, args.variant) for r in fit]
        mean, std = normalization(samples, args.variant)
        weights = np.array([r["sampling_weight"] for r in fit], np.float64)
        files = [args.data_root / r["files"][k] for r in records for k in sorted(r["files"])]
        signature = experiment_fingerprint(records, files, [], sources,
                       {"variant": args.variant, "seed": args.seed, "mean": mean.tolist(),
                        "std": std.tolist(), "epochs": epochs, "width": width, "depth": depth,
                        "batch": batch, "precision": "bf16", "augmentation": "C1-paired-gain"})
    order_template = np.repeat(np.arange(len(samples)), weights.astype(int))
    config = dict(spec_id="SPEC-91", seed=args.seed, variant=args.variant, smoke=args.smoke,
                  width=width, depth=depth, epochs=epochs, batch=batch, fingerprint=signature,
                  source_hashes={str(p): sha256_file(p) for p in sources},
                  data_manifest=None if args.smoke else sha256_file(args.manifest),
                  fit_ids=[r["sample_id"] for r in fit],
                  evaluation_ids=[r["sample_id"] for r in records if r["partition"] == "evaluation"],
                  mean=mean.tolist(), std=std.tolist(), software={"torch": torch.__version__,
                                                               "numpy": np.__version__})
    with (args.out / "data_manifest.json").open("x") as stream: json.dump(config, stream, indent=2)
    net = make_forest_model(args.variant, width, depth).cuda().train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(order_template) / batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                   total_steps=epochs * steps_per_epoch)
    # CPU resident dataset avoids a full pair cache consuming scarce VRAM.
    normalized = [torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))
                  for x, _, _ in samples]
    labels = [torch.from_numpy(y.astype(np.int64)) for _, y, _ in samples]
    del samples
    torch.cuda.reset_peak_memory_stats(); start = time.monotonic(); losses = []
    for epoch in range(epochs):
        order = rng.permutation(order_template); epoch_loss = []
        for pos in range(0, len(order), batch):
            idx = order[pos:pos + batch]
            if len(idx) == 1: idx = np.concatenate((idx, order[:1]))
            x = torch.stack([normalized[i] for i in idx]).cuda()
            y = torch.stack([labels[i] for i in idx]).cuda()
            x, y = augment_forest(x, y, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(x)
            loss = forest_loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training output")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("nonfinite gradient")
            optimizer.step(); scheduler.step(); epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
        if (epoch + 1) % 10 == 0 or args.smoke:
            print("epoch", epoch + 1, "loss", losses[-1], "seconds", round(time.monotonic() - start), flush=True)
    torch.save(dict(state=net.cpu().state_dict(), **config), args.out / "model.pt")
    result = dict(**config, loss=losses, seconds=time.monotonic() - start,
                  peak_vram_bytes=torch.cuda.max_memory_allocated(), completed=True)
    with (args.out / "summary.json").open("x") as stream: json.dump(result, stream, indent=2)
    print("COMPLETE", args.out, "smoke", args.smoke, flush=True)


if __name__ == "__main__": main()
