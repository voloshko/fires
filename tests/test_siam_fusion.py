"""SPEC-51: разностная фьюжн и двухэтапная потеря сиамской сети."""
import torch
from src.comp.hypothesis_models import SiameseUNet, two_stage_loss


def test_diff_fusion_forward_shape_and_fewer_params():
    x = torch.randn(2, 29, 64, 64)
    full, diff = SiameseUNet(8, 3, fusion="full"), SiameseUNet(8, 3, fusion="diff")
    assert full(x).shape == diff(x).shape == (2, 4, 64, 64)
    assert sum(p.numel() for p in diff.parameters()) < sum(p.numel() for p in full.parameters())


def test_two_stage_loss_prefers_correct_prediction():
    y = torch.tensor([[[0, 2], [3, 0]]])
    good = torch.full((1, 4, 2, 2), -3.0)
    for i in range(2):
        for j in range(2):
            good[0, y[0, i, j], i, j] = 3.0
    bad = -good
    assert torch.isfinite(two_stage_loss(good, y)) and two_stage_loss(good, y) < two_stage_loss(bad, y)


def test_two_stage_loss_without_burn_pixels():
    y = torch.zeros((1, 2, 2), dtype=torch.long)
    assert torch.isfinite(two_stage_loss(torch.randn(1, 4, 2, 2), y))


def test_soft_edge_targets_onehot_away_from_edges_and_split_at_edge():
    from src.comp.hypothesis_models import soft_edge_targets, soft_edge_loss
    y = torch.zeros((1, 8, 8), dtype=torch.long); y[0, :, 4:] = 2
    soft = soft_edge_targets(y, 3)
    assert torch.allclose(soft.sum(1), torch.ones(1, 8, 8))
    assert soft[0, 0, 2, 1] == 1.0 and soft[0, 2, 2, 6] == 1.0          # вдали от кромки — one-hot
    assert abs(soft[0, 0, 2, 3] - 2 / 3) < 1e-6 and abs(soft[0, 2, 2, 3] - 1 / 3) < 1e-6   # у кромки — доли
    good = torch.full((1, 4, 8, 8), -3.0); good[0, 0, :, :4] = 3.0; good[0, 2, :, 4:] = 3.0
    w = torch.tensor([.25, 1., 1., 1.])
    assert soft_edge_loss(good, y, 3, w) < soft_edge_loss(-good, y, 3, w)


def test_blob_loss_penalises_missed_small_component():
    from src.comp.hypothesis_models import blob_loss
    y = torch.zeros((1, 32, 32), dtype=torch.long); y[0, 2:6, 2:6] = 1; y[0, 10:30, 10:30] = 2     # малое и большое пятна
    good = torch.full((1, 4, 32, 32), -4.0); good[0, 0] = 4.0
    good[0, 1, 2:6, 2:6] = 4.0; good[0, 0, 2:6, 2:6] = -4.0; good[0, 2, 10:30, 10:30] = 4.0; good[0, 0, 10:30, 10:30] = -4.0
    miss_small = good.clone(); miss_small[0, 1, 2:6, 2:6] = -4.0; miss_small[0, 0, 2:6, 2:6] = 4.0
    assert blob_loss(good, y) < 0.1 and blob_loss(miss_small, y) > 0.4     # пропуск малого пятна — половина потери, несмотря на 16 пикс. из 1024
    assert torch.isfinite(blob_loss(good, torch.zeros((1, 32, 32), dtype=torch.long)))
