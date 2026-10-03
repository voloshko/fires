"""SPEC-93: matched 30-epoch HLS/L2A continuation of a C1-MM checkpoint."""
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
from src.comp.forest_pair import augment_forest, forest_loss, make_forest_model
from scripts.forest_pair_train import normalization
from src.comp.accuracy_artifacts import execution_sources, load_checkpoint, save_checkpoint


def load_twins(root, manifest):
    payload = json.loads(manifest.read_text())
    if not payload["ready_for_training"] or payload["qc_probe"]:
        raise ValueError("QC/provisional manifest cannot authorize training")
    records = payload["records"]; validate_manifest(records)
    for record in records:
        for key, path in record["files"].items():
            if sha256_file(root / path) != record["hashes"][key]: raise ValueError("twin source changed")
    return records


def read_twin(root, record, variant):
    with rasterio.open(root / record["files"][variant]) as raster: image = raster.read().astype(np.float32)
    with rasterio.open(root / record["files"]["mask"]) as raster: labels = raster.read(1).astype(np.int64)
    with rasterio.open(root / record["files"]["valid"]) as raster: valid = raster.read(1).astype(bool)
    valid &= np.isfinite(image).all(0) & (image != -9999).all(0) & (labels >= 0)
    if not valid.any(): raise ValueError("no common supervision")
    labels[~valid] = -1
    return np.where(valid[None], image, 0).astype(np.float32), labels, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--variant", choices=("hls", "l2a"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    if args.out.exists(): raise ValueError("immutable run directory already exists")
    records = load_twins(args.data_root, args.manifest)
    fit = [r for r in records if r["partition"] == "fit"]
    samples = [read_twin(args.data_root, r, args.variant) for r in fit]
    mean, std = normalization(samples, "post")
    bundle = load_checkpoint(args.checkpoint, sha256_file(args.checkpoint), export_root=Path("research/tensor-exports"))
    if bundle.get("seed") != args.seed or bundle.get("w") != 32 or bundle.get("cin", 6) != 6:
        raise ValueError("initializer is not the matching six-band C1-MM seed")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    net = make_forest_model("post", 32, 7); net.load_state_dict(bundle["state"]); del bundle
    net.cuda().train(); epochs, batch = 30, 8
    template = np.repeat(np.arange(len(fit)), [r["sampling_weight"] for r in fit])
    sources = execution_sources(Path(__file__), "src/comp/l2a_finetuning.py", "scripts/forest_pair_train.py", tta=False)
    inputs = [args.data_root / r["files"][k] for r in records for k in sorted(r["files"])]
    signature = experiment_fingerprint(records, inputs, [args.checkpoint], sources,
                dict(variant=args.variant, seed=args.seed, epochs=epochs, lr=3e-5, batch=batch,
                     mean=mean.tolist(), std=std.tolist(), scheduler="cosine", precision="bf16"))
    config = dict(spec_id="SPEC-93", variant=args.variant, seed=args.seed, epochs=epochs,
                  w=32, depth=7, cin=6, classes=2, mean=mean.tolist(), std=std.tolist(),
                  fingerprint=signature, data_manifest=sha256_file(args.manifest),
                  initialization_sha256=sha256_file(args.checkpoint),
                  initialization_family="C1-MM-seeds1-3",
                  fit_ids=[r["sample_id"] for r in fit], smoke=False,
                  source_hashes={str(p): sha256_file(p) for p in sources},
                  software={"torch": torch.__version__, "numpy": np.__version__})
    args.out.mkdir(parents=True)
    with (args.out / "data_manifest.json").open("x") as stream: json.dump(config, stream, indent=2)
    normalized = [torch.from_numpy(((x - mean[:, None, None]) / std[:, None, None]).astype(np.float32))
                  for x, _, _ in samples]
    labels = [torch.from_numpy(y) for _, y, _ in samples]; del samples
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * math.ceil(len(template) / batch))
    start = time.monotonic(); losses = []; torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        order = rng.permutation(template); loss_values = []
        for pos in range(0, len(order), batch):
            idx = order[pos:pos + batch]
            if len(idx) == 1: idx = np.concatenate((idx, order[:1]))
            x = torch.stack([normalized[i] for i in idx]).cuda()
            y = torch.stack([labels[i] for i in idx]).cuda()
            x, y = augment_forest(x, y, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(x)
            loss = forest_loss(logits, y)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss): raise FloatingPointError("nonfinite training")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step(); scheduler.step(); loss_values.append(float(loss.detach()))
        losses.append(float(np.mean(loss_values)))
        print("epoch", epoch + 1, "loss", losses[-1], "seconds", int(time.monotonic() - start), flush=True)
    save_checkpoint(args.out / "model.pt", net.cpu().state_dict(), config)
    result = dict(**config, completed=True, loss=losses, seconds=time.monotonic() - start,
                  peak_vram_bytes=torch.cuda.max_memory_allocated())
    with (args.out / "summary.json").open("x") as stream: json.dump(result, stream, indent=2)
    print("COMPLETE", args.variant, args.seed, flush=True)


if __name__ == "__main__": main()
