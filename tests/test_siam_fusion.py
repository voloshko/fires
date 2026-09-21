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
