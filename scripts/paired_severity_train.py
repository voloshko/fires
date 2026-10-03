"""SPEC-94: matched balanced/equal severity heads, without learning a boundary."""
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
from src.comp.accuracy_contract import experiment_fingerprint, sha256_file
from src.comp.forest_pair import augment_forest
from src.comp.paired_severity import PairedSeverityUNet, fit_class_weights, mtbs_targets, severity_loss
from scripts.forest_pair_train import load_frozen, normalization, read_sample
from src.comp.accuracy_artifacts import execution_sources, save_checkpoint


def severity_sample(root, record):
    image, _, valid = read_sample(root, record, "pair")
    with rasterio.open(root / record["files"]["severity"]) as raster: raw = raster.read(1)
    labels = mtbs_targets(raw, valid)
    if not (labels >= 0).any(): raise ValueError("sample has no severity targets")
    return image, labels, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--variant", choices=("balanced", "equal"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.out.exists(): raise ValueError("immutable run already exists")
    records = load_frozen(args.data_root, args.manifest)
    fit = [r for r in records if r["partition"] == "fit"]
    samples = [severity_sample(args.data_root, r) for r in fit]
    counts = sum((np.bincount(y[y >= 0], minlength=3) for _, y, _ in samples), np.zeros(3, np.int64))
    expected = np.asarray(json.loads(args.manifest.read_text())["fit_class_counts"])
    if not np.array_equal(counts, expected): raise ValueError("fit class counts differ from prepared manifest")
    weights = fit_class_weights(counts) if args.variant == "balanced" else np.ones(3, np.float32)
    mean, std = normalization(samples, "pair")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = PairedSeverityUNet().cuda().train(); epochs, batch = 120, 8
    sources = execution_sources(Path(__file__), "src/comp/paired_severity.py", "scripts/forest_pair_train.py", tta=False)
    files = [args.data_root / r["files"][k] for r in records for k in sorted(r["files"])]
    signature = experiment_fingerprint(records, files, [], sources,
                dict(variant=args.variant, seed=args.seed, mean=mean.tolist(), std=std.tolist(),
                     epochs=epochs, batch=batch, precision="bf16", class_weights=weights.tolist()))
    config = dict(spec_id="SPEC-94", variant=args.variant, seed=args.seed, width=32, depth=7,
                  epochs=epochs, batch=batch, classes=3, smoke=False, mean=mean.tolist(), std=std.tolist(),
                  fingerprint=signature, data_manifest=sha256_file(args.manifest),
                  fit_class_counts=counts.tolist(), class_weights=weights.tolist(),
                  fit_ids=[r["sample_id"] for r in fit],
                  evaluation_ids=[r["sample_id"] for r in records if r["partition"] == "evaluation"],
                  source_hashes={str(p): sha256_file(p) for p in sources},
                  software={"torch": torch.__version__, "numpy": np.__version__})
    args.out.mkdir(parents=True)
    with (args.out / "data_manifest.json").open("x") as stream: json.dump(config, stream, indent=2)
    normalized = [torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))
                  for x, _, _ in samples]
    labels = [torch.from_numpy(y) for _, y, _ in samples]; del samples
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                                   total_steps=epochs * math.ceil(len(fit) / batch))
    losses = []; start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        order = rng.permutation(len(fit)); values = []
        for pos in range(0, len(order), batch):
            idx = order[pos:pos + batch]
            if len(idx) == 1: idx = np.concatenate((idx, order[:1]))
            x = torch.stack([normalized[i] for i in idx]).cuda()
            y = torch.stack([labels[i] for i in idx]).cuda()
            x, y = augment_forest(x, y, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(x)
            loss = severity_loss(logits, y, weights)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError("nonfinite training")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step(); scheduler.step(); values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
        if (epoch + 1) % 10 == 0:
            print("epoch", epoch + 1, "loss", losses[-1], "seconds", int(time.monotonic() - start), flush=True)
    save_checkpoint(args.out / "model.pt", net.cpu().state_dict(), config)
    result = dict(**config, completed=True, loss=losses, seconds=time.monotonic() - start,
                  peak_vram_bytes=torch.cuda.max_memory_allocated())
    with (args.out / "summary.json").open("x") as stream: json.dump(result, stream, indent=2)
    print("COMPLETE", args.variant, args.seed, flush=True)


if __name__ == "__main__": main()
