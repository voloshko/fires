"""SPEC-111: a trainable temporal correction around an immutable post model."""
import torch
from torch import nn
import torch.nn.functional as F


def expand_probability(probability, addition):
    if probability.shape != addition.shape or probability.numel() == 0:
        raise ValueError("aligned nonempty probabilities required")
    if (not torch.isfinite(probability).all() or not torch.isfinite(addition).all()
            or (probability < 0).any() or (probability > 1).any()
            or (addition < 0).any() or (addition > 1).any()):
        raise ValueError("invalid reference/addition probabilities")
    p, q = probability.float(), addition.float()
    return p+(1-p)*q


def expansion_log_probs(original, value):
    base = original.float().log_softmax(1)
    z = value.float()
    background = base[:, 0]+F.logsigmoid(-z)
    foreground = torch.logaddexp(base[:, 1], base[:, 0]+F.logsigmoid(z))
    return torch.stack((background, foreground), 1)


class RecallExpansion(nn.Module):
    def __init__(self, base, variant="pair", rank=8):
        super().__init__()
        if variant not in ("pair", "post") or rank != 8:
            raise ValueError("fixed pair/post adapter with rank8 required")
        if base.down[0][0].in_channels != 6 or base.head.out_channels != 2:
            raise ValueError("frozen six-band binary C1-MM required")
        self.base, self.variant = base, variant
        self.base.requires_grad_(False)
        self.base.eval()
        widths = [stage[0].out_channels for stage in base.down]
        self.projections = nn.ModuleList([nn.Sequential(nn.Conv2d(3*w, rank, 1), nn.ReLU()) for w in widths])
        self.correction_head = nn.Sequential(nn.Conv2d(rank*len(widths), 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.constant_(self.correction_head[-1].bias, -4)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Freeze BN statistics as well as parameters.
        return self

    def _encode(self, x):
        skips = []
        for i, stage in enumerate(self.base.down):
            x = stage(x if i == 0 else self.base.pool(x))
            skips.append(x)
        return skips

    def forward_with_addition(self, x):
        factor = 2**(len(self.base.down)-1)
        if (x.ndim != 4 or x.shape[0] == 0 or x.shape[1] != 12
                or min(x.shape[-2:]) < factor or any(s % factor for s in x.shape[-2:])
                or not torch.isfinite(x).all()):
            raise ValueError("finite nonempty pre6/post6 frame required")
        with torch.no_grad():
            post = self._encode(x[:, 6:])
            out = post[-1]
            for k, (up, conv) in enumerate(zip(self.base.up, self.base.conv)):
                out = conv(torch.cat((up(out), post[-2-k]), 1))
            original = self.base.head(out)
            pre = self._encode(x[:, :6]) if self.variant == "pair" else post
        projected = []
        for p, a, b in zip(self.projections, pre, post):
            feature = p(torch.cat((a, b, b-a), 1))
            projected.append(F.interpolate(feature, size=x.shape[-2:], mode="bilinear", align_corners=False))
        value = self.correction_head(torch.cat(projected, 1))[:, 0].float()
        logits = expansion_log_probs(original, value)
        return logits, value.sigmoid()

    def forward(self, x):
        return self.forward_with_addition(x)[0]

    def adapter_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith("base.")}

    def load_adapter_state(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("base.")}
        if set(state) != expected:
            raise ValueError("adapter-only state differs or contains a backbone")
        self.load_state_dict(state, strict=False)


def validate_expansion_binding(bundle, base_sha256, variant):
    if (bundle.get("spec_id") != "SPEC-111" or bundle.get("base_sha256") != base_sha256
            or bundle.get("variant") != variant or bundle.get("rank") != 8):
        raise ValueError("adapter base/variant binding differs")
