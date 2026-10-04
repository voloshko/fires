"""SPEC-109: full-fit exposure, fixed base identities and artifact budgets."""
import hashlib
import json
from pathlib import Path
import shutil
import torch

from .accuracy_contract import canonical_hash, sha256_file

BASE_MANIFEST_SHA = "1a70bf29a0b9e3c8462396967fdc12fca7f92ae29a236636de5585c74692f373"
RUSSIAN_MANIFEST_SHA = "1d487c711beb657a4c0ea0114379ddc51be2a31302891e47699942f95459f3db"
BASE_SHAS = (
    "03751038fbf6406d0db4e40727d2f6fb562cf26b570ccf880a32c945f6ece19c",
    "a0282b7c31c77d0941c1ed0d10981dd526c7345c5c7335cd2d8fb071ec40c0e0",
    "48a3faa63c8814f793e2963f8db8ef2a42466d0f021463f84e58b5cdb834d61e",
)
OUTPUT_BUDGET = 512*1024**2
FREE_FLOOR = 3*1024**3


def fit_population(base, russian):
    fit = [r for r in base if r["partition"] == "fit"]
    reserve = [r for r in russian if r["partition"] == "fit"]
    if len(fit) != 374 or len(reserve) != 309:
        raise ValueError("frozen fit population differs")
    rows = fit+reserve
    if len({r["sample_id"] for r in rows}) != 683:
        raise ValueError("duplicate fit identity")
    return rows


def exposure_pool(records):
    pool = []
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("empty/duplicate exposure identities")
    for i, record in enumerate(records):
        weight = record["sampling_weight"]
        if record["partition"] != "fit" or isinstance(weight, bool) or weight not in (1, 2):
            raise ValueError("only fit observations and original integer repetition allowed")
        pool.extend([i]*int(weight))
    return pool


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(json.dumps([key, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def guard_output(root, reserve=2*1024**2):
    root = Path(root)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    if used+reserve > OUTPUT_BUDGET or shutil.disk_usage(root if root.exists() else root.parent).free-reserve < FREE_FLOOR:
        raise ValueError("adapter output budget/free-space reserve unavailable")


def read_seal(path):
    seal = json.loads(Path(path).read_text())
    if canonical_hash({k: v for k, v in seal.items() if k != "recipe_id"}) != seal["recipe_id"]:
        raise ValueError("training recipe content changed")
    for name, digest in seal["source_hashes"].items():
        if sha256_file(name) != digest:
            raise ValueError("sealed training source changed")
    return seal
