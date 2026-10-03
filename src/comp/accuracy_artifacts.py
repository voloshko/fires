"""SPEC-95: restricted legacy loading, tensor exports and matched recipes."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.torch_version import TorchVersion

from src.comp.accuracy_contract import canonical_hash, sha256_file


def execution_sources(*paths, tta=True):
    base = ["src/comp/forest_pair.py", "scripts/train_unet.py", "src/comp/accuracy_contract.py",
            "src/comp/accuracy_artifacts.py"]
    if tta: base.append("src/comp/hls.py")
    return [Path(p) for p in sorted(set(base + [str(p) for p in paths]))]


def _json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.tolist()
    if isinstance(value, dict): return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_json_value(v) for v in value]
    if isinstance(value, str): return str(value)
    if value is None or type(value) in (bool, int, float): return value
    raise ValueError(f"unsupported checkpoint metadata type {type(value).__name__}")


def restricted_legacy(path, expected_sha256):
    if sha256_file(path) != expected_sha256: raise ValueError("raw checkpoint checksum mismatch")
    allowed = [TorchVersion, np.ndarray, np.dtype,
               np._core.multiarray._reconstruct, np.dtypes.Float32DType, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_checkpoint(path, expected_sha256, export_root=None):
    """Never invoke unrestricted pickle, even for a known local legacy artifact."""
    path = Path(path)
    if sha256_file(path) != expected_sha256: raise ValueError("raw checkpoint checksum mismatch")
    native_metadata = path.with_name(path.stem + ".metadata.json")
    if native_metadata.exists():
        stored = json.loads(native_metadata.read_text())
        if stored.get("schema_version") != 2 or stored["state_sha256"] != expected_sha256:
            raise ValueError("native artifact provenance mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        digest = bytes(payload["metadata_digest"].tolist()).hex()
        if digest != canonical_hash(stored["metadata"]): raise ValueError("native metadata differs from tensor-bound digest")
        state = payload["state"]
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("native state is not tensor-only")
        return dict(stored["metadata"], state=state)
    if export_root is None:
        tensor_path = path.with_name(path.stem + ".tensor.pt")
        meta_path = path.with_name(path.stem + ".tensor.json")
    else:
        destination = Path(export_root) / expected_sha256
        destination.mkdir(parents=True, exist_ok=True)
        tensor_path, meta_path = destination / "weights.pt", destination / "metadata.json"
    if tensor_path.exists() or meta_path.exists():
        if not tensor_path.exists() or not meta_path.exists(): raise ValueError("incomplete tensor export")
        stored = json.loads(meta_path.read_text())
        if stored.get("schema_version") != 1: raise ValueError("unsupported tensor export schema")
        if (stored["source_sha256"] != expected_sha256 or stored["state_sha256"] != sha256_file(tensor_path)
                or stored["metadata_sha256"] != canonical_hash(stored["metadata"])):
            raise ValueError("tensor export provenance mismatch")
        state = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("tensor export contains non-tensor state")
        raw = restricted_legacy(path, expected_sha256)
        actual_metadata = _json_value({k: v for k, v in raw.items() if k != "state"})
        if (canonical_hash(actual_metadata) != stored["metadata_sha256"] or state.keys() != raw["state"].keys()
                or any(v.dtype != raw["state"][k].dtype or v.shape != raw["state"][k].shape
                       or not torch.equal(v, raw["state"][k]) for k, v in state.items())):
            raise ValueError("tensor export does not represent its declared raw source")
        return dict(stored["metadata"], state=state)
    raw = restricted_legacy(path, expected_sha256)
    if not isinstance(raw, dict) or "state" not in raw:
        raise ValueError("legacy checkpoint must contain state and metadata")
    state = raw["state"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("legacy state contains non-tensors")
    metadata = _json_value({k: v for k, v in raw.items() if k != "state"})
    canonical_hash(metadata)  # Reject NaN and unsupported metadata before exporting.
    with tensor_path.open("xb") as stream: torch.save(state, stream)
    stored = dict(schema_version=1, source_sha256=expected_sha256, state_sha256=sha256_file(tensor_path),
                  metadata_sha256=canonical_hash(metadata), metadata=metadata)
    with meta_path.open("x") as stream: json.dump(stored, stream, indent=2)
    return dict(metadata, state=state)


def assert_matched_recipes(metadata, fields):
    if len(metadata) < 2: raise ValueError("at least two runs needed for matched recipe audit")
    common = ("source_hashes", "software", "data_manifest", "fit_ids") + tuple(fields)
    first = metadata[0]
    for field in common:
        if field not in first: raise ValueError(f"missing recipe field {field}")
        for other in metadata[1:]:
            if other.get(field) != first[field]: raise ValueError(f"matched recipe drift in {field}")
    return canonical_hash({key: first[key] for key in common})


def checkpoint_metadata(path, export_root=None):
    path = Path(path)
    bundle = load_checkpoint(path, sha256_file(path), export_root=export_root)
    metadata = {k: v for k, v in bundle.items() if k != "state"}
    archive = path.with_name("data_manifest.json")
    if archive.exists() and canonical_hash(json.loads(archive.read_text())) != canonical_hash(metadata):
        raise ValueError("run archive differs from checkpoint-bound metadata")
    return metadata


def freeze_recipe(path, sources, configuration):
    """Append-once seal; subsequent starts must use precisely the same code/config."""
    payload = dict(source_hashes={str(p): sha256_file(p) for p in sources},
                   configuration=configuration,
                   software={"torch": str(torch.__version__), "numpy": str(np.__version__)})
    payload["recipe_id"] = canonical_hash(payload)
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload: raise ValueError("sealed recipe changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream: json.dump(payload, stream, indent=2)
    return payload["recipe_id"]


def save_checkpoint(path, state, metadata):
    """New models store one tensor artifact, cryptographically binding JSON metadata."""
    path = Path(path); meta_path = path.with_name(path.stem + ".metadata.json")
    if path.exists() or meta_path.exists(): raise ValueError("model artifact already exists")
    if not all(isinstance(v, torch.Tensor) for v in state.values()): raise ValueError("state must contain tensors only")
    metadata = _json_value(metadata); digest = canonical_hash(metadata)
    with path.open("xb") as stream:
        torch.save(dict(state=state, metadata_digest=torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)), stream)
    payload = dict(schema_version=2, state_sha256=sha256_file(path), metadata=metadata)
    with meta_path.open("x") as stream: json.dump(payload, stream, indent=2)
