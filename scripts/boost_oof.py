"""SPEC-48: карты вероятностей бустинга для канала сети, без утечки.

Для фолда f: обучающие чипы получают out-of-fold вероятности (3 внутренних
групповых фолда по fire_event_id), оценочные — вероятности бустинга на всей
обучающей части (уже посчитаны в research/bs-confirm-boost-f{f}-swir-v1).
Запускать с FEATURES=swir.
"""
import sys, json, hashlib, argparse, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path
from src.comp.features import stack, NAMES
from src.comp.model import build_training_set, train
from src.comp.hypothesis_lab import bs_confirmation_split, write_json
from scripts.train_unet import load_split
p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, required=True); p.add_argument('--inner', type=int, default=3); p.add_argument('--boost', default='research/bs-confirm-boost-f{f}-swir-v1'); p.add_argument('--out', default='research/boost_oof-f{f}'); a = p.parse_args()
assert 'mirbi' in ' '.join(NAMES), 'нужен FEATURES=swir'
d, fit, tune = load_split('data/comp/train/bs', 'data/comp/split_bs.json', use_all=False)
fit, tune = bs_confirmation_split(d.meta, fit, tune, a.fold)
out = Path(a.out.format(f=a.fold)); out.mkdir(parents=True, exist_ok=True)
ev = {c: e for c, e in d.meta.set_index('chip_id')['fire_event_id'].to_dict().items()}
events = sorted({str(ev[c]) for c in fit}, key=lambda e: hashlib.sha256(f'oof:{e}'.encode()).hexdigest())
inner = {e: i % a.inner for i, e in enumerate(events)}
hashes = {}
for k in range(a.inner):
    tr = [c for c in fit if inner[str(ev[c])] != k]; te = [c for c in fit if inner[str(ev[c])] == k]
    x, y = build_training_set(d, tr); m = train(x, y)
    for c in te:
        chip = d.load(c); f = np.nan_to_num(stack(chip), posinf=0, neginf=0)
        pr = m.predict_proba(f.reshape(len(f), -1).T).reshape(*chip.shape, 4).astype(np.float16); np.save(out / f'{c}.npy', pr); hashes[c] = hashlib.sha256(pr.tobytes()).hexdigest()[:16]
    print(f'фолд {a.fold} внутренний {k}: обучено на {len(tr)}, карты для {len(te)}', flush=True)
bd = Path(a.boost.format(f=a.fold)); ids = json.load(open(bd / 'data_manifest.json'))['evaluation']; assert ids == tune
P = np.load(bd / 'probabilities.npy')
for c, pr in zip(ids, P): np.save(out / f'{c}.npy', pr.astype(np.float16)); hashes[c] = hashlib.sha256(pr.astype(np.float16).tobytes()).hexdigest()[:16]
write_json(out / 'manifest.json', dict(fold=a.fold, inner=a.inner, features=list(NAMES), fit=fit, evaluation=tune, source_eval=str(bd), sha256_16=hashes))
print('готово', len(hashes), 'карт →', out)
