import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.comp.accuracy_artifacts import assert_matched_recipes, execution_sources, freeze_recipe, load_checkpoint
from src.comp.accuracy_contract import canonical_hash, experiment_fingerprint, sha256_file


def test_legacy_numpy_and_tensor_metadata_roundtrip(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(dict(state={"weight": torch.ones(2, 3)}, mean=np.array([1, 2], np.float32),
                    std=[1, 1], software={"torch": torch.__version__}), path)
    bundle = load_checkpoint(path, sha256_file(path))
    assert bundle["mean"] == [1, 2] and torch.equal(bundle["state"]["weight"], torch.ones(2, 3))
    other = load_checkpoint(path, sha256_file(path))
    assert torch.equal(bundle["state"]["weight"], other["state"]["weight"])


def test_export_and_raw_tamper_rejected(tmp_path):
    path = tmp_path / "model.pt"; torch.save(dict(state={"w": torch.ones(2)}), path)
    original = sha256_file(path); load_checkpoint(path, original)
    tensor = path.with_name("model.tensor.pt"); tensor.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="provenance"): load_checkpoint(path, original)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"): load_checkpoint(path, original)


def test_metadata_tamper_rejected(tmp_path):
    path = tmp_path / "model.pt"; torch.save(dict(state={"w": torch.ones(2)}, epochs=120), path)
    expected = sha256_file(path); load_checkpoint(path, expected)
    meta = path.with_name("model.tensor.json"); x = json.loads(meta.read_text()); x["metadata"]["epochs"] = 1
    meta.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="provenance"): load_checkpoint(path, expected)
    x["metadata_sha256"] = canonical_hash(x["metadata"]); meta.write_text(json.dumps(x))
    with pytest.raises(ValueError, match="declared raw source"): load_checkpoint(path, expected)


def test_arbitrary_pickle_is_never_executed(tmp_path):
    marker = tmp_path / "executed"
    class Payload:
        def __reduce__(self): return eval, (f"open({str(marker)!r},'w').write('unsafe')",)
    path = tmp_path / "unsafe.pt"; torch.save(dict(state={"w": torch.ones(2)}, payload=Payload()), path)
    with pytest.raises(Exception): load_checkpoint(path, sha256_file(path))
    assert not marker.exists()


def test_matched_recipe_allows_seed_and_declared_variant_only():
    base = dict(source_hashes={"a": "hash"}, software={"torch": "version"}, data_manifest="data",
                fit_ids=["fit-a"], epochs=120, seed=1, variant="post")
    other = dict(base, seed=2, variant="pair")
    assert_matched_recipes([base, other], fields=("epochs",))
    changed = copy.deepcopy(other); changed["source_hashes"]["a"] = "new"
    with pytest.raises(ValueError, match="source_hashes"):
        assert_matched_recipes([base, changed], fields=("epochs",))


def test_model_dependency_is_in_fingerprint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    paths = execution_sources()
    assert Path("scripts/train_unet.py") in paths
    for p in paths: p.parent.mkdir(parents=True, exist_ok=True); p.write_text("original")
    before = experiment_fingerprint([], [], [], paths, {})
    Path("scripts/train_unet.py").write_text("changed forward")
    assert before != experiment_fingerprint([], [], [], paths, {})


def test_recipe_seal_cannot_follow_mutable_code(tmp_path):
    source = tmp_path / "train.py"; source.write_text("original")
    recipe = tmp_path / "recipe.json"
    before = freeze_recipe(recipe, [source], {"epochs": 30})
    assert before == freeze_recipe(recipe, [source], {"epochs": 30})
    source.write_text("new recipe")
    with pytest.raises(ValueError, match="sealed recipe changed"):
        freeze_recipe(recipe, [source], {"epochs": 30})
