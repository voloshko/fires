import numpy as np
import pytest
import torch

from src.comp.paired_severity import (
    PairedSeverityUNet, apply_fixed_boundary, fit_class_weights, mtbs_targets, severity_loss,
)


def test_mtbs_background_and_unknown_never_become_low_severity():
    raw = np.array([[0, 1, 2, 3, 4, 5, 6, 255]], np.uint8)
    known = np.ones_like(raw, bool)
    np.testing.assert_array_equal(mtbs_targets(raw, known), [[-1, -1, 0, 1, 2, -1, -1, -1]])
    known[0, 2] = False
    assert mtbs_targets(raw, known)[0, 2] == -1


def test_rare_fit_classes_get_more_weight():
    weights = fit_class_weights([1800, 200, 100])
    assert weights[0] < weights[1] < weights[2] <= 4
    np.testing.assert_allclose(weights.mean(), 1, atol=1e-6)
    with pytest.raises(ValueError): fit_class_weights([100, 0, 10])


def test_severity_loss_masks_both_terms():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    labels = torch.randint(0, 3, (2, 8, 8)); labels[:, :4] = -1
    changed = x.detach().clone(); changed[:, :, :4] = -100
    a = severity_loss(x, labels, [1, 1, 1]); b = severity_loss(changed, labels, [1, 1, 1])
    assert torch.allclose(a, b)
    a.backward(); assert x.grad[:, :, :4].abs().sum() == 0
    assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError): severity_loss(x, torch.full_like(labels, -1), [1, 1, 1])


def test_severity_cannot_expand_fixed_boundary():
    p = np.zeros((3, 8, 8), np.float32); p[2] = 1
    boundary = np.zeros((8, 8), bool); boundary[:4] = True
    valid = np.ones_like(boundary); valid[0] = False
    predicted = apply_fixed_boundary(p, boundary, valid)
    assert np.all(predicted[1:4] == 3) and np.all(predicted[4:] == 0)
    assert np.all(predicted[0] == 0)


def test_three_class_pair_model_backward_and_reload(tmp_path):
    torch.set_num_threads(2)
    net = PairedSeverityUNet(width=4, depth=3)
    x = torch.randn(2, 12, 32, 32, requires_grad=True)
    logits = net(x); assert logits.shape == (2, 3, 32, 32)
    severity_loss(logits, torch.randint(0, 3, (2, 32, 32)), [1, 1, 1]).backward()
    assert x.grad[:, :6].abs().sum() > 0 and x.grad[:, 6:].abs().sum() > 0
    net.eval(); path = tmp_path / "weights.pt"; torch.save(net.state_dict(), path)
    loaded = PairedSeverityUNet(width=4, depth=3).eval()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad(): assert torch.equal(net(x), loaded(x))
