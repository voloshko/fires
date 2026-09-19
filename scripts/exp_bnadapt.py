"""Адаптация BatchNorm на входах теста (test-time adaptation) для оптических
сетей: статистики BN пересчитываются по 89 тестовым чипам (без меток), веса не
меняются. Замер честный: адаптируем по ТЕСТОВЫМ входам, меряем на 35 настроечных
— если помогает, значит сдвиг распределения тест/обучение реален. Контроль:
адаптация по самим 35 настроечным (оптимистичный верх)."""
import sys, json, hashlib, numpy as np, torch; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from src.comp.chips import BsDataset
from src.comp.features import stack
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far
from src.comp.unet import load as load_net
TAGS = ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1')
z = np.load('models/tune_proba_19.npz'); PB, T, OK = z['pb'].astype(np.float32), z['t'], z['ok']
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
tune = sorted(sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest())[:35]); chips = [d.load(c) for c in tune]
ZERO = np.stack([c.label_zero() for c in chips]); test = BsDataset('data/comp/test/bs'); test_chips = [test.load(c) for c in test.chip_ids()]
def feats(model, ch):
    net, mean, std, dev = model
    f = np.nan_to_num(stack(ch, net.names), posinf=0, neginf=0).astype(np.float32)
    return torch.from_numpy((f - mean[:, None, None]) / std[:, None, None]).unsqueeze(0).to(dev)
def probs(model, ch):
    net, *_ = model; x = feats(model, ch)
    with torch.no_grad():
        lg = net(x).float()
        for dims in ([2], [3], [2, 3]): lg = lg + torch.flip(net(torch.flip(x, dims)).float(), dims)
        return (lg / 4).softmax(1)[0].permute(1, 2, 0).cpu().numpy()
def adapt(model, src_chips):
    net = model[0]
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None   # накопительное среднее
    net.train()
    with torch.no_grad():
        for ch in src_chips: net(feats(model, ch))
    net.eval()
def measure(pn, name):
    P = 0.4*PB + 0.6*pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
    r = score_bs_micro(list(T), list(out)); print(f'{name:40s} {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f}', flush=True)
base = np.mean([np.load(f'models/exp_{t}.tune.npy').astype(np.float32) for t in TAGS], 0); measure(base, 'без адаптации (кэш)')
for src_name, src in (('адаптация по 89 тестовым', test_chips), ('адаптация по 35 настроечным (верх)', chips)):
    acc = []
    for t in TAGS:
        m = load_net(f'models/exp_{t}.pt'); adapt(m, src); acc.append(np.stack([probs(m, c) for c in chips]))
    measure(np.mean(acc, 0), src_name)
