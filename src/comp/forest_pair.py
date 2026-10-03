"""SPEC-91: a six-band shared date encoder with stable fusion."""
import torch
from torch import nn
import torch.nn.functional as F
from scripts.train_unet import UNet, block


class ForestPairUNet(nn.Module):
    def __init__(self, width=32, depth=7):
        super().__init__()
        if depth < 2: raise ValueError("depth must be at least two")
        widths = [width * 2 ** i for i in range(depth)]
        self.down = nn.ModuleList([block(6 if i == 0 else widths[i - 1], w)
                                   for i, w in enumerate(widths)])
        self.pool = nn.MaxPool2d(2)
        self.fuse = nn.ModuleList([nn.Sequential(nn.Conv2d(3 * w, w, 1, bias=False),
                                                 nn.BatchNorm2d(w), nn.ReLU(inplace=True))
                                   for w in widths])
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i - 1], 2, 2)
                                 for i in range(depth - 1, 0, -1)])
        self.conv = nn.ModuleList([block(2 * widths[i - 1], widths[i - 1])
                                   for i in range(depth - 1, 0, -1)])
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 12: raise ValueError("expected pre6 + post6")
        pre, post = x[:, :6], x[:, 6:]
        skips = []
        for i, (encode, fuse) in enumerate(zip(self.down, self.fuse)):
            n = len(pre); dates = encode(torch.cat((pre, post), dim=0))
            pre, post = dates[:n], dates[n:]
            skips.append(fuse(torch.cat((pre, post, post - pre), dim=1)))
            if i < len(self.down) - 1: pre, post = self.pool(pre), self.pool(post)
        out = skips[-1]
        for i, (up, conv) in enumerate(zip(self.up, self.conv)):
            out = conv(torch.cat((up(out), skips[-i - 2]), dim=1))
        return self.head(out)


def make_forest_model(variant, width=32, depth=7):
    if variant == "pair": return ForestPairUNet(width, depth)
    if variant == "post": return UNet(6, classes=2, w=width, depth=depth)
    raise ValueError(variant)


def forest_loss(logits, labels):
    known = labels >= 0
    if not known.any(): raise ValueError("batch has no supervision")
    weights = logits.new_tensor([.25, 1.], dtype=torch.float32)
    ce = F.cross_entropy(logits.float(), labels, weight=weights, ignore_index=-1)
    prob = logits.float().softmax(1)[:, 1][known]
    truth = (labels[known] == 1).float()
    return ce + 1 - (2 * (prob * truth).sum() + 1) / (prob.sum() + truth.sum() + 1)


def augment_forest(x, y, rng, gain=.1):
    if rng.random() < .5: x, y = x.flip(3), y.flip(2)
    if rng.random() < .5: x, y = x.flip(2), y.flip(1)
    k = int(rng.integers(4)); x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (1, 2))
    g = torch.as_tensor(rng.uniform(1 - gain, 1 + gain, (len(x), 6, 1, 1)),
                        device=x.device, dtype=x.dtype)
    if x.shape[1] == 12: g = g.repeat(1, 2, 1, 1)
    elif x.shape[1] != 6: raise ValueError("expected six or twelve channels")
    return x * g, y
