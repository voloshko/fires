"""SPEC-121: uniform per-member residuals and lossless compressed final masks."""
import io
from pathlib import Path
import shutil

import numpy as np
import torch

from .accuracy_contract import bind_cache, verify_cache
from .temporal_adapter_evaluation import pack_prediction, unpack_prediction

BUDGET, FLOOR = 16*1024**2, 3*1024**3


def member_delta(probability, addition, members=5):
    if (members != 5 or probability.shape != addition.shape
            or not torch.isfinite(probability).all() or not torch.isfinite(addition).all()
            or (probability < 0).any() or (probability > 1).any()
            or (addition < 0).any() or (addition > 1).any()):
        raise ValueError('five aligned member probabilities required')
    return (1-probability.float())*addition.float()/5


def anchored_probability(anchor, reproduced_anchor, delta):
    a, b, d = (np.asarray(v, np.float32) for v in (anchor, reproduced_anchor, delta))
    if (a.shape != b.shape or a.shape != d.shape or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.isfinite(d).all()
            or (a < 0).any() or (a > 1).any() or (b < 0).any() or (b > 1).any()
            or (d < 0).any() or (d > 1).any()):
        raise ValueError('invalid aligned anchor/delta')
    if not np.allclose(a,b,atol=1e-6,rtol=1e-5):
        raise ValueError('original C1-MM5 reference parity failed')
    result=a+d
    if result.max() > 1+2e-5:
        raise ValueError('member residual exceeded numerical probability bound')
    return np.clip(result,0,1)


def guard_output(directory, reserve=0):
    root=Path(directory)
    used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if reserve<0 or used+reserve>BUDGET or shutil.disk_usage(root).free-reserve<FLOOR:
        raise ValueError('coupled cache budget/free-floor unavailable')


def save_mask(path, prediction, signature):
    path=Path(path);stream=io.BytesIO()
    np.savez_compressed(stream,mask=pack_prediction(prediction))
    content=stream.getvalue();guard_output(path.parent,len(content)+4096)
    with path.open('xb')as target:target.write(content)
    bind_cache(path,signature)


def load_mask(path, signature, shape):
    verify_cache(path,signature)
    with np.load(path,allow_pickle=False)as source:
        if set(source.files)!={'mask'}:raise ValueError('compressed final-mask payload differs')
        return unpack_prediction(source['mask'],shape)
