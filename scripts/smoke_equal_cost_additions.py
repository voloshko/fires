"""SPEC-114: real512 mechanical GPU check, never an accuracy measurement."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet import UNet
from scripts.forest_pair_train import read_sample
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe, load_checkpoint, save_checkpoint
from src.comp.accuracy_contract import sha256_file
from src.comp.equal_cost_additions import LOSS_ID, equal_cost_loss as forest_loss
from src.comp.recall_expansion import RecallExpansion, expand_probability
from src.comp.equal_cost_additions import validate_equal_cost_binding as validate_expansion_binding

BASE_SHA = "03751038fbf6406d0db4e40727d2f6fb562cf26b570ccf880a32c945f6ece19c"
MANIFEST_SHA = "1d487c711beb657a4c0ea0114379ddc51be2a31302891e47699942f95459f3db"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".pt").exists():
        raise ValueError("immutable smoke already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("real GPU preflight requires CUDA")
    manifest = args.data_root/"russia/manifest.json"
    if sha256_file(manifest) != MANIFEST_SHA or sha256_file(args.base) != BASE_SHA:
        raise ValueError("wrong frozen smoke input/base")
    records = json.loads(manifest.read_text())["records"]
    record = sorted((r for r in records if r["partition"] == "fit"), key=lambda r: r["sample_id"])[0]
    for key, relative in record["files"].items():
        if sha256_file(args.data_root/relative) != record["hashes"][key]:
            raise ValueError("smoke source pixels changed")
    sources = execution_sources(Path(__file__), "src/comp/recall_expansion.py", "src/comp/equal_cost_additions.py", "scripts/forest_pair_train.py",
        "scripts/queues/smoke_equal_cost_additions.sh")
    recipe = freeze_recipe(args.out.with_suffix(".recipe.json"), sources,
        dict(base_sha256=BASE_SHA, manifest_sha256=MANIFEST_SHA, sample_id=record["sample_id"],
            hashes=record["hashes"], rank=8, variants=["pair", "post"], smoke=True))
    torch.manual_seed(1)
    bundle = load_checkpoint(args.base, BASE_SHA, export_root=Path("research/tensor-exports"))
    base = UNet(6, classes=2, w=bundle.get("w", bundle.get("width", 32)), depth=7)
    base.load_state_dict(bundle["state"])
    mean, std = np.asarray(bundle["mean"], np.float32), np.asarray(bundle["std"], np.float32)
    del bundle
    image, labels, _ = read_sample(args.data_root, record, "pair")
    image = ((image-np.tile(mean, 2)[:, None, None])/np.tile(std, 2)[:, None, None]).astype(np.float32)
    x = torch.from_numpy(image)[None].cuda()
    y = torch.from_numpy(labels)[None].cuda()
    base = base.cuda().eval()
    rows = []
    for variant in ("pair", "post"):
        model = RecallExpansion(base, variant).cuda().train()
        parity = {}
        for name, enabled in (("fp32", False), ("bf16", True)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                expected = base(x[:, 6:]).float()
                actual, delta = model.forward_with_addition(x)
            reference = expected.softmax(1)[:, 1]
            expanded = expand_probability(reference, delta)
            if not torch.allclose(actual.exp()[:, 1], expanded, atol=1e-6, rtol=1e-5):
                raise ValueError("real512 probability formula failed: "+name)
            if ((reference >= .5) & (expanded < .5)).any():
                raise ValueError("real512 reference positives lost")
            parity[name] = True
        bn = {k: v.clone() for k, v in base.named_buffers()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = forest_loss(logits, y)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite smoke loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):
            raise ValueError("missing/nonfinite smoke correction gradients")
        optimizer.step()
        if any(p.grad is not None for p in base.parameters()) or any(not torch.equal(v, dict(base.named_buffers())[k]) for k, v in bn.items()):
            raise ValueError("frozen backbone/BN changed")
        checkpoint = args.out.with_name(args.out.stem+"-"+variant).with_suffix(".pt")
        if shutil.disk_usage(checkpoint.parent).free < 3*1024**3:
            raise ValueError("checkpoint free-space floor unavailable")
        state = model.adapter_state()
        save_checkpoint(checkpoint, state, dict(spec_id="SPEC-114", loss_id=LOSS_ID, base_sha256=BASE_SHA,
            variant=variant, rank=8, recipe_id=recipe, smoke=True, fit_ids=[record["sample_id"]],
            mean=mean.tolist(), std=std.tolist()))
        restored = load_checkpoint(checkpoint, sha256_file(checkpoint))
        validate_expansion_binding(restored, BASE_SHA, variant)
        for key, tensor in state.items():
            if not torch.equal(restored["state"][key], tensor):
                raise ValueError("native smoke adapter state differs")
        rows.append(dict(variant=variant, probability_formula=parity, reference_positives_preserved=True, gradients_finite_nonzero=True,
            backbone_bn_frozen=True, checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
        del model, optimizer, restored, state, gradients, logits, loss
    result = dict(spec_id="SPEC-114", recipe_id=recipe, base_sha256=BASE_SHA,
        sample_id=record["sample_id"], variants=rows, completed=True,
        non_claims=["Mechanical GPU/source/adapter check only, no full training or accuracy gain.",
            "No new independent competitor comparison, severity/L2A useful gain or goal completion."])
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
