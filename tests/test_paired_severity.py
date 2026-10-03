import numpy as np
import pytest
import torch

from src.comp.paired_severity import (
    PairedSeverityUNet, apply_fixed_boundary, fit_class_weights, mtbs_targets, severity_loss,
    grouped_severity_comparison, mtbs_evaluation_labels, severity_confusion, severity_metrics,
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


def test_evaluation_background_is_not_ignored():
    raw = np.array([0, 1, 2, 3, 4, 5, 6], np.uint8)
    labels = mtbs_evaluation_labels(raw, np.ones_like(raw, bool))
    np.testing.assert_array_equal(labels, [0, 0, 1, 2, 3, 0, -1])
    cm = severity_confusion(labels, np.array([1, 2, 1, 2, 3, 3, 3]))
    assert cm[0, 1:].sum() == 3 and cm[1:, 1:].sum() == 3
    assert severity_metrics(cm)["miou"] == .5


def test_severity_bootstrap_uses_whole_events():
    good = np.diag([10, 10, 10, 10])
    weak = np.array([[10, 0, 0, 0], [0, 5, 5, 0], [0, 0, 5, 5], [0, 5, 0, 5]])
    result = grouped_severity_comparison(["a", "a", "b"], [good] * 3, [weak] * 3, n_boot=50)
    aggregated = grouped_severity_comparison(["a", "b"], [2 * good, good], [2 * weak, weak], n_boot=50)
    assert result["ci95"] == aggregated["ci95"] and result["groups"] == 2
    assert result["delta"] > 0 and result["ci95"][0] > 0
    with pytest.raises(ValueError): severity_metrics(np.zeros((4, 4)))
