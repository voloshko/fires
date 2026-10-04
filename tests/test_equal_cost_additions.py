import pytest
import torch
import torch.nn.functional as F

from src.comp.equal_cost_additions import LOSS_ID, equal_cost_loss, validate_equal_cost_binding
from src.comp.forest_pair import forest_loss


def test_equal_ce_and_unchanged_dice_formula():
    logits = torch.tensor([[[[.2, -.7, 2.]], [[.8, .4, -1.]]]], requires_grad=True)
    labels = torch.tensor([[[0, 1, -1]]])
    probability = logits.softmax(1)[:, 1][labels >= 0]
    truth = (labels[labels >= 0] == 1).float()
    expected = F.cross_entropy(logits, labels, ignore_index=-1) + 1 - (
        2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    actual = equal_cost_loss(logits, labels)
    assert torch.allclose(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, :, 2]) == 0


def test_background_pressure_increases_relative_to_foreground():
    labels = torch.tensor([[[0, 1]]])
    ratios = []
    for loss in (forest_loss, equal_cost_loss):
        logits = torch.zeros((1, 2, 1, 2), requires_grad=True)
        loss(logits, labels).backward()
        ratios.append(float(abs(logits.grad[0, 1, 0, 0] / logits.grad[0, 1, 0, 1])))
    assert ratios[1] > ratios[0]


def test_unknown_only_batch_rejected():
    with pytest.raises(ValueError, match="no supervision"):
        equal_cost_loss(torch.zeros((1, 2, 2, 2)), torch.full((1, 2, 2), -1))


def test_spec_loss_and_base_binding():
    good = dict(spec_id="SPEC-114", loss_id=LOSS_ID, base_sha256="base", variant="pair", rank=8)
    validate_equal_cost_binding(good, "base", "pair")
    for key, value in (("spec_id", "SPEC-111"), ("loss_id", "other"),
                       ("base_sha256", "wrong"), ("variant", "post"), ("rank", 16)):
        with pytest.raises(ValueError, match="binding differs"):
            validate_equal_cost_binding({**good, key: value}, "base", "pair")
