"""Prithvi-EO-2.0-300M (NASA/IBM, Apache 2.0) как кодировщик (research 3.2).
Предобучен на HLS с теми же шестью полосами, что есть в нашем стеке Sentinel-2
(B02 B03 B04 B8A B11 B12), на 4 кадрах во времени — наша пара «до/после» идёт
двумя кадрами. Декодер лёгкий: токены последнего и среднего слоя → 32×32 →
свёртки с четырёхкратным повышением до 512. Дообучение целиком, малый шаг у
кодировщика. Замер — на тех же 35 чипах; вероятности кэшируются как у сетей."""
import os, sys, json, hashlib, time, numpy as np; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import torch, torch.nn as nn, torch.nn.functional as F
from terratorch.registry import BACKBONE_REGISTRY
from src.comp.chips import BsDataset
from src.comp.metric import score_bs_micro
from src.comp.postproc import drop_far

DEV = 'cuda'; SEED = int(os.environ.get('SEED', 20260918)); EPOCHS = int(os.environ.get('EPOCHS', 30)); TAG = os.environ.get('TAG', 'prithvi')
LR_ENC = float(os.environ.get('LR_ENC', 5e-5)); LR_DEC = 1e-3; BATCH = 2; BANDS = (0, 1, 2, 6, 7, 8)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
d = BsDataset('data/comp/train/bs'); s = json.load(open('data/comp/split_bs.json')); ids = [c for c in s['train'] if d.has_post(c)]
rank = sorted(ids, key=lambda c: hashlib.sha256(f'tune:{c}'.encode()).hexdigest()); tune, fit = sorted(rank[:35]), sorted(rank[35:])
def tensor(ch):
    pre, post = ch.pre[list(BANDS)].astype(np.float32) / 10000, ch.post[list(BANDS)].astype(np.float32) / 10000
    return np.stack([pre, post], 1)   # (6, 2, H, W)
t0 = time.time()
xtr = [tensor(d.load(c)) for c in fit]; ytr = [d.load(c).mask.astype(np.int64) for c in fit]
chips_t = [d.load(c) for c in tune]; xva = [tensor(c) for c in chips_t]; T = np.stack([c.mask for c in chips_t]); OK = np.stack([c.valid() for c in chips_t]); ZERO = np.stack([c.label_zero() for c in chips_t])
mean = np.mean([x.mean((1, 2, 3)) for x in xtr], 0); std = np.mean([x.std((1, 2, 3)) for x in xtr], 0) + 1e-6
X = ((torch.stack([torch.from_numpy(x) for x in xtr]) - torch.tensor(mean).view(1, -1, 1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1, 1)).half().to(DEV)
Y = torch.stack([torch.from_numpy(y) for y in ytr]).to(DEV)
print(f'[{TAG}] данные на карте за {time.time()-t0:.0f}с: {tuple(X.shape)}', flush=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = BACKBONE_REGISTRY.build('prithvi_eo_v2_300', pretrained=True, bands=['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2'], num_frames=2, img_size=512)
        self.proj = nn.Conv2d(2 * 2 * 1024, 256, 1)
        def up(cin, cout): return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.dec = nn.Sequential(up(256, 128), up(128, 64), up(64, 32), up(32, 32))
        self.skip = nn.Sequential(nn.Conv2d(12, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 4, 1)
    def tokens(self, t):   # (B, 2049, 1024) → (B, 2*1024, 32, 32)
        b = t.shape[0]; t = t[:, 1:].reshape(b, 2, 32, 32, 1024).permute(0, 1, 4, 2, 3).reshape(b, 2 * 1024, 32, 32); return t
    def forward(self, x):
        feats = self.enc(x); f = torch.cat([self.tokens(feats[-1]), self.tokens(feats[len(feats) // 2])], 1)
        h = self.dec(self.proj(f)); sk = self.skip(x.flatten(1, 2))
        return self.head(torch.cat([h, sk], 1))

net = Net().to(DEV)
opt = torch.optim.AdamW([{'params': net.enc.parameters(), 'lr': LR_ENC}, {'params': [p for n, p in net.named_parameters() if not n.startswith('enc.')], 'lr': LR_DEC}], weight_decay=1e-4)
steps = EPOCHS * (len(fit) // BATCH); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [LR_ENC, LR_DEC], total_steps=steps)
weight = torch.tensor([0.25, 1, 1, 1.], device=DEV); scaler = torch.amp.GradScaler(DEV)
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    net.train(); order = rng.permutation(len(fit)); tot = 0
    for k in range(0, len(order) - BATCH + 1, BATCH):
        idx = torch.as_tensor(order[k:k + BATCH], device=DEV); x, y = X[idx].float(), Y[idx]
        if rng.random() < 0.5: x, y = x.flip(4), y.flip(2)
        if rng.random() < 0.5: x, y = x.flip(3), y.flip(1)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEV):
            out = net(x); loss = F.cross_entropy(out, y, weight=weight)
            pb = 1 - out.float().softmax(1)[:, 0]; tb = (y > 0).float()
            loss = loss + 1 - (2 * (pb * tb).sum() + 1) / (pb.sum() + tb.sum() + 1)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched.last_epoch < steps - 1: sched.step()
        tot += float(loss)
    if ep % 5 == 0 or ep == EPOCHS: print(f'[{TAG}] эпоха {ep} loss {tot/(len(order)//BATCH):.4f} ({time.time()-t0:.0f}с)', flush=True)
net.eval(); probs = []
with torch.no_grad(), torch.amp.autocast(DEV):
    for x in xva:
        xt = ((torch.from_numpy(x) - torch.tensor(mean).view(-1, 1, 1, 1)) / torch.tensor(std).view(-1, 1, 1, 1)).float().unsqueeze(0).to(DEV)
        lg = net(xt).float()
        for dims in ([4], [3], [3, 4]):
            lg = lg + torch.flip(net(torch.flip(xt, dims)).float(), [dd - 1 for dd in dims])
        probs.append((lg / 4).softmax(1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))
PN = np.stack(probs); np.save(f'models/exp_{TAG}.tune.npy', PN)
torch.save({'state': net.state_dict(), 'mean': mean, 'std': std, 'bands': BANDS, 'arch': 'prithvi_eo_v2_300'}, f'models/exp_{TAG}.pt')
z = np.load('models/tune_proba_19.npz'); PB = z['pb'].astype(np.float32)
def measure(pn, name):
    P = 0.4 * PB + 0.6 * pn; burn = P.argmax(3) > 0; burn[~OK] = (pn.argmax(3) > 0)[~OK]
    out = np.where(burn, P[..., 1:].argmax(3) + 1, 0).astype(np.uint8); out[ZERO] = 0; out = np.stack([drop_far(o) for o in out])
    r = score_bs_micro(list(T), list(out)); print(f'[{TAG}] {name}: {r["iou_burn"]:.4f}/{r["miou_sev"]:.4f} кл1 {r["per_class"][1]:.3f} взв {(0.35*r["iou_burn"]+0.30*r["miou_sev"])/0.65:.4f}', flush=True)
pn1 = PN.astype(np.float32); measure(pn1, 'Prithvi одна + бустинг')
base = [f'models/exp_{t}.tune.npy' for t in ('d7w32', 'd7s1', 'd7s2', 'd7fast', 'd7rot', 'd7lov', 'd7lov_s1', 'd7bnd', 'd7bnd_s1')]
nine = np.mean([np.load(f).astype(np.float32) for f in base], 0); measure(nine, 'девятка'); measure((9 * nine + pn1) / 10, 'девятка + Prithvi')
print(f'[{TAG}] обучено за {time.time()-t0:.0f}с')
