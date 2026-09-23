import numpy as np, torch
from src.comp.hls import tile_holdout, augment, tta8

def test_holdout_by_tile():
    names = [f'subsetted_512x512_HLS.S30.T{t:02d}ABC.20{y}001.v1.4_merged.tif' for t in range(60) for y in (18, 19, 20)]
    h = tile_holdout(names)
    for i in range(0, len(names), 3): assert h[i] == h[i + 1] == h[i + 2]
    assert 0.05 < np.mean(h) < 0.4

def test_augment_keeps_alignment():
    rng = np.random.default_rng(0); y = torch.zeros(2, 8, 8, dtype=torch.long); y[:, 1, 2] = 1
    x = y[:, None].float().repeat(1, 3, 1, 1)
    for _ in range(20):
        xa, ya = augment(x, y, rng, rot=True, gain=0.1)
        assert torch.equal((xa[:, 0] > 0), ya == 1)

def test_tta8_identity_for_equivariant_net():
    net = lambda x: x[:, :2] * 2
    x = torch.randn(1, 3, 8, 8); assert torch.allclose(tta8(net, x), x[:, :2] * 2, atol=1e-6)

def test_scorer_iou_and_compare():
    from src.comp.hls_eval import Scorer
    Y = np.zeros((4, 8, 8), bool); Y[:, :4] = True; V = np.ones_like(Y)
    s = Scorer(Y, V, n_boot=50); perfect, half = s.stats(Y.astype(np.float32), 0.5), s.stats((Y & (np.arange(8) < 2)[None, None]).astype(np.float32), 0.5)
    assert s.iou(perfect) == 1.0 and abs(s.iou(half) - 0.25) < 1e-9   # 4 строки × 2 столбца из 4 × 8
    assert s.compare(perfect, half)[3] == 'ЛУЧШЕ'
