"""SPEC-118: matched full-pool training of small temporal/post corrections."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet import UNet
from scripts.forest_pair_train import load_frozen, read_sample
from scripts.russian_transfer_train import load_russian
from src.comp.accuracy_contract import canonical_hash, sha256_file, validate_manifest
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe, load_checkpoint, save_checkpoint
from src.comp.forest_pair import augment_forest
from src.comp.equal_cost_additions import LOSS_ID, equal_cost_loss as forest_loss
from src.comp.recall_expansion import RecallExpansion
from src.comp.five_seed_additions import FIVE_BASE_SHAS as BASE_SHAS, PARENT_RECIPE, PARENT_RESULT_SHA, audit_parent
from src.comp.temporal_adapter_contract import (
    BASE_MANIFEST_SHA, RUSSIAN_MANIFEST_SHA, exposure_pool,
    fit_population, guard_output, read_seal, tensor_digest,
)


def seal_configuration(root, reference_root):
    base = root/"forest-pairs/manifest.json"
    russian = root/"russia/manifest.json"
    if sha256_file(base) != BASE_MANIFEST_SHA or sha256_file(russian) != RUSSIAN_MANIFEST_SHA:
        raise ValueError("frozen source manifests differ")
    _, _, _, parent_hashes = audit_parent(root)
    for seed, digest in enumerate(BASE_SHAS, 1):
        if sha256_file(reference_root/f"hls-c1mm-final-s{seed}/model.pt") != digest:
            raise ValueError("frozen initializer differs")
    return dict(base_manifest=BASE_MANIFEST_SHA, russian_manifest=RUSSIAN_MANIFEST_SHA,
        initializers=list(BASE_SHAS), parent_recipe=PARENT_RECIPE, parent_result=PARENT_RESULT_SHA, parent_adapters=parent_hashes, epochs=30, batch=8, rank=8, lr=.001, weight_decay=.0001,
        seeds=[4, 5], variants=["pair", "post"], precision="bf16", normalization="locked-C1-MM",
        fit_unique=683, full_pool=True, loss_id=LOSS_ID, ce_weights=[1, 1], addition_prior_logit=-4, scoring="C1-MM5+uniform-mean5-probability-union")


def training_sources():
    return execution_sources("scripts/train_five_seed_additions.py", "src/comp/recall_expansion.py", "src/comp/equal_cost_additions.py", "src/comp/five_seed_additions.py", "scripts/score_equal_cost_additions.py", "scripts/train_equal_cost_additions.py",
        "src/comp/temporal_adapter_contract.py", "scripts/forest_pair_train.py",
        "scripts/russian_transfer_train.py", "src/comp/russian_transfer.py",
        "scripts/prepare_russian_windows.py", "src/comp/l2a_finetuning.py",
        "scripts/queues/queue_five_seed_additions.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data-root", "reference-root", "runs"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--seal-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=(4, 5))
    parser.add_argument("--variant", choices=("pair", "post"))
    args = parser.parse_args()
    configuration = seal_configuration(args.data_root, args.reference_root)
    args.runs.mkdir(parents=True, exist_ok=True)
    guard_output(args.runs)
    recipe_id = freeze_recipe(args.runs/"recipe.json", training_sources(), configuration)
    if args.seal_only:
        print("SEALED", recipe_id, flush=True)
        return
    if args.seed is None or args.variant is None or not torch.cuda.is_available():
        raise ValueError("full GPU training requires seed and variant")
    destination = args.runs/f"{args.variant}-s{args.seed}"
    if destination.exists():
        raise ValueError("immutable training run already exists")
    base = load_frozen(args.data_root, args.data_root/"forest-pairs/manifest.json")
    russian = load_russian(args.data_root, args.data_root/"russia/manifest.json")
    validate_manifest(base+russian)
    fit = fit_population(base, russian)
    pool = exposure_pool(fit)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    initializer = args.reference_root/f"hls-c1mm-final-s{args.seed}/model.pt"
    bundle = load_checkpoint(initializer, BASE_SHAS[args.seed-1], export_root=Path("research/tensor-exports"))
    backbone = UNet(6, classes=2, w=bundle.get("w", bundle.get("width", 32)), depth=7)
    backbone.load_state_dict(bundle["state"])
    mean, std = np.asarray(bundle["mean"], np.float32), np.asarray(bundle["std"], np.float32)
    del bundle
    model = RecallExpansion(backbone, args.variant).cuda().train()
    base_digest = tensor_digest(model.base.state_dict())
    images, labels = [], []
    for record in fit:
        x, y, _ = read_sample(args.data_root, record, "pair")
        normalized = ((x-np.tile(mean, 2)[:, None, None])/np.tile(std, 2)[:, None, None]).astype(np.float32)
        images.append(torch.from_numpy(normalized))
        labels.append(torch.from_numpy(y))
    epochs, batch = 30, 8
    steps = math.ceil(len(pool)/batch)
    metadata = dict(spec_id="SPEC-118", loss_id=LOSS_ID, base_sha256=BASE_SHAS[args.seed-1], variant=args.variant,
        rank=8, seed=args.seed, epochs=epochs, batch=batch, steps_per_epoch=steps, smoke=False,
        mean=mean.tolist(), std=std.tolist(), recipe_id=recipe_id, width=32, depth=7,
        fit_ids=[r["sample_id"] for r in fit], sampling_weights=[r["sampling_weight"] for r in fit],
        base_manifest=BASE_MANIFEST_SHA, russian_manifest=RUSSIAN_MANIFEST_SHA,
        source_hashes=read_seal(args.runs/"recipe.json")["source_hashes"],
        software=dict(torch=str(torch.__version__), numpy=str(np.__version__)))
    guard_output(args.runs)
    destination.mkdir()
    with (destination/"data_manifest.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*steps)
    start = time.monotonic()
    exposures = Counter()
    traces, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    print("TRAIN_START", args.variant, args.seed, "unique", len(fit), "views_per_epoch", len(pool), flush=True)
    for epoch in range(epochs):
        read_seal(args.runs/"recipe.json")
        order = rng.permutation(pool).tolist()
        traces.append(canonical_hash([fit[i]["sample_id"] for i in order]))
        values = []
        for offset in range(0, len(order), batch):
            indices = order[offset:offset+batch]
            x = torch.stack([images[i] for i in indices]).cuda()
            y = torch.stack([labels[i] for i in indices]).cuda()
            x, y = augment_forest(x, y, rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = forest_loss(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite adapter loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite adapter gradient")
            if any(p.grad is not None for p in model.base.parameters()):
                raise ValueError("backbone received gradient")
            optimizer.step()
            scheduler.step()
            exposures.update(fit[i]["sample_id"] for i in indices)
            values.append(float(loss.detach()))
        losses.append(float(np.mean(values)))
        print("EPOCH", epoch+1, "loss", losses[-1], "seconds", round(time.monotonic()-start, 1), flush=True)
    if tensor_digest(model.base.state_dict()) != base_digest:
        raise ValueError("backbone/BN state changed")
    expected = {r["sample_id"]: epochs*int(r["sampling_weight"]) for r in fit}
    if dict(exposures) != expected:
        raise ValueError("original fit exposure differs")
    read_seal(args.runs/"recipe.json")
    guard_output(args.runs)
    save_checkpoint(destination/"adapter.pt", model.adapter_state(), metadata)
    summary = dict(completed=True, spec_id="SPEC-118", recipe_id=recipe_id,
        variant=args.variant, seed=args.seed, losses=losses, epoch_order_hashes=traces,
        actual_exposures=dict(exposures), base_tensor_digest=base_digest, backbone_unchanged=True,
        seconds=time.monotonic()-start, peak_vram_bytes=torch.cuda.max_memory_allocated())
    with (destination/"summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    print("COMPLETE", args.variant, args.seed, flush=True)


if __name__ == "__main__":
    main()
