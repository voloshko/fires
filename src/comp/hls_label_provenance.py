"""SPEC-115: exact source-label reuse is distinct from event attribution."""
from datetime import date
import hashlib
import numpy as np

from .accuracy_contract import canonical_hash


def label_signature(labels, valid, grid):
    labels, valid = np.asarray(labels), np.asarray(valid)
    if (labels.ndim != 2 or labels.shape != valid.shape or valid.dtype != bool
            or not np.isin(labels, [-1, 0, 1]).all()):
        raise ValueError("aligned binary labels and boolean validity required")
    return canonical_hash(dict(grid=grid, shape=list(labels.shape),
        labels_sha256=hashlib.sha256(labels.astype('<i2').tobytes()).hexdigest(),
        valid_sha256=hashlib.sha256(np.packbits(valid, bitorder='little').tobytes()).hexdigest()))


def prior_ignition(ignition, acquisition):
    if ignition is None:
        return False
    return date.fromisoformat(ignition) <= date.fromisoformat(acquisition)


def reuse_witness(record):
    return dict(sample_id=record['sample_id'], acquisition_day=record['acquisition_day'],
                existing_registry_event=record.get('event_id'),
                attribution='identical target/domain only; not a physical burn date')
