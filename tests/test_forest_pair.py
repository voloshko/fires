import io
import numpy as np
import pytest
import torch

from src.comp.forest_pair import ForestPairUNet, augment_forest, forest_loss, make_forest_model


def test_both_dates_receive_gradients():
    torch.set_num_threads(2)
    net = ForestPairUNet(width=4, depth=3)
    x = torch.randn(2, 12, 32, 32, requires_grad=True)
    logits = net(x)
    assert logits.shape == (2, 2, 32, 32)
    forest_loss(logits, torch.randint(0, 2, (2, 32, 32))).backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :6].abs().sum() > 0 and x.grad[:, 6:].abs().sum() > 0


def test_loss_ignores_pixels_in_ce_and_dice():
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    y = torch.ones(2, 8, 8, dtype=torch.long); y[:, :4] = -1
    changed = logits.detach().clone(); changed[:, :, :4] = 100
    assert torch.allclose(forest_loss(logits, y), forest_loss(changed, y))
    forest_loss(logits, y).backward()
    assert logits.grad[:, :, :4].abs().sum() == 0
    with pytest.raises(ValueError): forest_loss(logits, torch.full_like(y, -1))


def test_augmentation_preserves_pair_and_label_alignment():
    y = torch.zeros(2, 16, 16, dtype=torch.long); y[:, 3, 5] = 1
    x = y[:, None].float().repeat(1, 12, 1, 1)
    xa, ya = augment_forest(x, y, np.random.default_rng(0))
    assert torch.equal(xa[:, :6], xa[:, 6:])
    assert torch.equal(xa[:, 0] > 0, ya > 0)


def test_save_load_prediction_identity():
    net = ForestPairUNet(width=4, depth=3).eval(); x = torch.randn(1, 12, 32, 32)
    with torch.no_grad(): before = net(x)
    stream = io.BytesIO(); torch.save(net.state_dict(), stream); stream.seek(0)
    loaded = ForestPairUNet(width=4, depth=3).eval()
    loaded.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad(): assert torch.equal(before, loaded(x))


def test_post_control_and_wrong_input():
    net = make_forest_model("post", width=4, depth=3).eval()
    assert net(torch.randn(1, 6, 32, 32)).shape == (1, 2, 32, 32)
    with pytest.raises(ValueError): ForestPairUNet(width=4, depth=3)(torch.randn(1, 6, 32, 32))
