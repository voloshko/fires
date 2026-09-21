"""Isolated experiment primitives (SPEC-32/33/36); no competition test access."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

OPTICAL = ('dnbr','rbr','nbr_pre','nbr_post','ndvi_pre','ndvi_post','dndvi','nbr2_post','b12_post','b8a_post','landcover')
import os as _os
if _os.environ.get('FEATURES') == 'swir':   # SPEC-47: SWIR-индексы в оптическом наборе
    OPTICAL = OPTICAL + ('mirbi_pre', 'mirbi_post', 'dmirbi', 'nbr2_pre', 'dnbr2', 'b11_post')
GRID = np.unique(np.round(np.r_[np.arange(.5,.95,.05), np.arange(.95,1,.005), .999], 6))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''): h.update(block)
    return h.hexdigest()


def rank_ids(ids, salt):
    return sorted(ids, key=lambda c: hashlib.sha256(f'{salt}:{c}'.encode()).hexdigest())


def nested_folds(ids, k=5):
    """Match legacy outer AF folds; calibration is internal to outer fit."""
    ids = sorted(ids)
    if len(set(ids)) != len(ids) or len(ids) < k*2: raise ValueError('nonunique or insufficient IDs')
    for j in range(k):
        test = ids[j::k]
        dev = rank_ids([c for i,c in enumerate(ids) if i%k != j], f'lab-cal:{j}')
        n = max(1, round(.2*len(dev)))
        cal, fit = dev[:n], dev[n:]
        assert not (set(fit)&set(cal) or set(fit)&set(test) or set(cal)&set(test))
        yield sorted(fit), sorted(cal), test


def choose_background(background, rng, scores=None, budget=1500):
    background = np.asarray(background)
    if len(background) <= budget: return background.copy()
    if scores is None: return rng.choice(background, budget, replace=False)
    n = budget//2
    hard = background[np.argsort(scores[background], kind='stable')[-n:]]
    rest = np.setdiff1d(background, hard, assume_unique=True)
    return np.r_[hard, rng.choice(rest,budget-n,replace=False)]


def binary_counts(truth, prediction):
    truth, prediction = np.asarray(truth,dtype=bool), np.asarray(prediction,dtype=bool)
    return np.array([np.count_nonzero(truth&prediction),np.count_nonzero(~truth&prediction),np.count_nonzero(truth&~prediction)],dtype=np.int64)


def binary_scores(counts):
    tp,fp,fn = np.asarray(counts).sum(axis=0) if np.asarray(counts).ndim > 1 else counts
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=float(tp/max(1,tp+fp)),recall=float(tp/max(1,tp+fn)),f1=float(2*tp/max(1,2*tp+fp+fn)))


def paired_interval(a,b,seed=20260918,n=2000):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3: raise ValueError('expected matching chip TP FP FN')
    rng=np.random.default_rng(seed); delta=[]
    for _ in range(n):
        ix=rng.integers(len(a),size=len(a))
        delta.append(binary_scores(b[ix])['f1']-binary_scores(a[ix])['f1'])
    return list(map(float,np.quantile(delta,[.025,.975])))


def extra_channels(chip, optical):
    """SPEC-48/49: дополнительные входные каналы по переменным окружения.
    EXTRA_CHANNELS_DIR — каталог карт {chip_id}.npy (H, W, C), например вероятности
    бустинга (auto-context); LOCAL_Z=1 — локальная z-оценка dNBR и dMIRBI в окне 31.
    Без переменных возвращает вход как есть."""
    import os
    from scipy.ndimage import uniform_filter
    from src.comp.features import stack
    parts=[optical]
    d=os.environ.get('EXTRA_CHANNELS_DIR')
    if d:
        m=np.load(f"{d}/{chip.chip_id}.npy").astype(np.float32)
        parts.append(np.transpose(m,(2,0,1)) if m.ndim==3 and m.shape[-1]<=8 else m)
    if os.environ.get('LOCAL_Z')=='1':
        for a in stack(chip,('dnbr','dmirbi')):
            mu=uniform_filter(a,31,mode='nearest'); sd=np.sqrt(np.maximum(uniform_filter(a*a,31,mode='nearest')-mu*mu,0))
            parts.append(((a-mu)/(sd+1e-3))[None])
    return optical if len(parts)==1 else np.concatenate(parts).astype(np.float32)


def bs_inputs(chip, variant):
    from src.comp.features import stack
    optical=stack(chip,OPTICAL)
    if variant=='optical': optical=extra_channels(chip,optical)
    if variant=='optical': return optical
    if not chip.post.size: raise ValueError('paired experiment requires post scene')
    raw=np.concatenate([chip.pre[:9],chip.post[:9]]).astype(np.float32)/10000
    if variant in ('raw','siam'): return np.concatenate([raw,optical])
    raise ValueError(variant)


def confusion(truth,prediction,classes=4,valid=None):
    t,p=np.asarray(truth).ravel(),np.asarray(prediction).ravel()
    if valid is not None: t,p=t[np.asarray(valid).ravel()],p[np.asarray(valid).ravel()]
    if np.any((t<0)|(t>=classes)|(p<0)|(p>=classes)): raise ValueError('invalid class')
    return np.bincount(t*classes+p,minlength=classes**2).reshape(classes,classes)


def bs_scores(cm):
    cm=np.asarray(cm); tp=cm[1:,1:].sum(); fp=cm[0,1:].sum(); fn=cm[1:,0].sum()
    den=cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou=[float(cm[i,i]/den[i]) if den[i]>0 else None for i in range(len(cm))]
    burn=float(tp/(tp+fp+fn)) if tp+fp+fn else None
    present=[v for v in iou[1:] if v is not None]
    sev=float(np.mean(present)) if present else None
    weighted=(.35*burn+.30*sev)/.65 if burn is not None and sev is not None else None
    return dict(iou_burn=burn,miou_sev=sev,weighted=weighted,per_class=iou)


def write_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def bs_confirmation_split(meta,development,selection,fold,k=5):
    """Confirm on original 144 fit chips; the known 35 selection chips never score."""
    if fold not in range(k): raise ValueError('invalid fold')
    mapping=meta.set_index('chip_id')['fire_event_id'].to_dict()
    used=development+selection
    if len(set(used))!=len(used): raise ValueError('duplicate chip')
    if any(c not in mapping or str(mapping[c]).lower() in ('','nan','none') for c in used): raise ValueError('missing event identity')
    groups={mapping[c] for c in development}; selected={mapping[c] for c in selection}
    if groups&selected: raise ValueError('selection/development event overlap')
    held=set(rank_ids(groups,'bs-confirm')[fold::k])
    evaluation=sorted(c for c in development if mapping[c] in held)
    fit=sorted(selection+[c for c in development if mapping[c] not in held])
    return fit,evaluation


def bs_prediction(network,boost,chip,weight=.6):
    """Canonical product SCL/severity/far-fire rules for cached probabilities."""
    from src.comp.postproc import drop_far
    mixed=network if boost is None else weight*network+(1-weight)*boost
    pred=mixed.argmax(2).astype(np.uint8)
    blind=~chip.valid()
    pred[blind]=np.where(network.argmax(2)>0,mixed[...,1:].argmax(2)+1,0)[blind]
    pred[chip.label_zero()]=0
    return drop_far(pred,125)


def verify_bs_probability_cache(root,fit,evaluation):
    """Bind a boost cache to exact splits and source pixels before scoring."""
    root=Path(root); manifest=json.loads((root/'data_manifest.json').read_text())
    if manifest['fit']!=fit or manifest['evaluation']!=evaluation or set(fit)&set(evaluation):
        raise ValueError('boost cache split mismatch')
    if not manifest['files']: raise ValueError('boost cache has no source hashes')
    for filename,expected in manifest['files'].items():
        if digest(filename)!=expected: raise ValueError('boost cache source changed')
    path=root/'probabilities.npy'
    return {str(path):digest(path),str(root/'data_manifest.json'):digest(root/'data_manifest.json')}


def harmonize_s2_dn(values,bands,processing_baseline):
    """ESA PB >=04.00 adds 1000 DN; convert to harmonized DN, never fit a scene."""
    if processing_baseline is None: raise ValueError('missing processing baseline')
    baseline=float(processing_baseline)
    if not np.isfinite(baseline) or baseline<0: raise ValueError('invalid processing baseline')
    out=np.asarray(values,dtype=np.float32).copy()
    if baseline>=4:
        for i,band in enumerate(bands):
            if band!='SCL': out[i]=np.maximum(out[i]-1000,0)
    return out
