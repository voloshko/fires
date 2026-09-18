import sys, json, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.af import (AfDataset, build_training_set, train, predict_model,
                         predict_threshold, score_chips, save, f1, features, NAMES)
t0=time.time()
d = AfDataset('data/comp/train/af'); sp = json.load(open('data/comp/split_af.json'))
x, y = build_training_set(d, sp['train'])
print('пикселей', x.shape, 'горящих', int(y.sum()), f'{time.time()-t0:.0f}с')
m = train(x, y); print(f'обучено за {time.time()-t0:.0f}с')
# решающая граница подбирается на ВАЛИДАЦИОННОЙ части
F,Y=[],[]
for c in sp['val']:
    ch=d.load(c); F.append(features(ch).reshape(len(NAMES),-1).T); Y.append(ch.mask.reshape(-1)>0)
proba=m.predict_proba(np.concatenate(F))[:,1]; Y=np.concatenate(Y)
best=max(((f1(Y, proba>=c)['f1'], c) for c in np.arange(0.05,0.96,0.05)))
print(f'граница по VAL: {best[1]:.2f}, F1_val={best[0]:.4f}')
save(m, 'models/af_hgb.pkl', cutoff=float(best[1]))
r = score_chips(d, sp['holdout'], lambda ch: predict_model(m, ch, best[1]))
b = json.load(open('data/comp/baseline_af.json'))
print('МОДЕЛЬ holdout', {k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()})
print(f'BASELINE был: F1={b["f1"]:.4f} prec={b["precision"]:.4f} rec={b["recall"]:.4f}')
json.dump({'cutoff':float(best[1]),'n_train_chips':len(sp['train']),**r}, open('data/comp/model_af.json','w'), indent=2)
