import copy
import unittest
import tempfile
from pathlib import Path

import torch

from scripts.train_unet import UNet
from src.comp.accuracy_artifacts import save_checkpoint, load_checkpoint
from src.comp.accuracy_contract import sha256_file
from src.comp.temporal_adapter import TemporalAdapter, bounded_correction, odds_update, validate_adapter_binding


class TemporalAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)
        self.base = UNet(6, classes=2, w=4, depth=3).eval()
        self.x = torch.randn(2, 12, 16, 16)

    def test_zero_head_matches_post_and_backbone_bn_stays_frozen(self):
        model = TemporalAdapter(copy.deepcopy(self.base)).train()
        original = model.base(self.x[:, 6:])
        state = {k: v.clone() for k, v in model.base.state_dict().items()}
        output = model(self.x)
        self.assertTrue(torch.equal(original, output))
        changed = self.x.clone()
        changed[:, :6] *= -8
        self.assertTrue(torch.equal(output, model(changed)))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertTrue(all(torch.equal(v, model.base.state_dict()[k]) for k, v in state.items()))
        self.assertGreater(float(model.correction_head[-1].weight.grad.abs().sum()), 0)

    def test_nonzero_pair_head_uses_pre_date_control_does_not(self):
        changed = self.x.clone()
        changed[:, :6] += 4
        for variant in ("pair", "post"):
            model = TemporalAdapter(copy.deepcopy(self.base), variant).eval()
            torch.nn.init.constant_(model.correction_head[-1].weight, .1)
            a = model.forward_with_correction(self.x)[1]
            b = model.forward_with_correction(changed)[1]
            self.assertEqual(torch.equal(a, b), variant == "post")

    def test_correction_bounds_odds_endpoints_and_confidence(self):
        raw = torch.tensor([-100., -1., 0., 1., 100.], requires_grad=True)
        delta = bounded_correction(raw)
        self.assertTrue((delta >= -2).all() and (delta <= 4).all())
        delta.sum().backward()
        self.assertGreater(float(raw.grad[2]), 0)  # Zero initialization must learn.
        p = torch.tensor([0., .25, .5, .75, 1.])
        self.assertTrue(torch.equal(odds_update(p, torch.zeros_like(p)), p))
        self.assertEqual(float(odds_update(p, delta)[0].detach()), 0)
        self.assertEqual(float(odds_update(p, delta)[-1].detach()), 1)
        self.assertGreater(float(odds_update(torch.tensor([.9]), torch.tensor([-2.]))[0]), .5)

    def test_adapter_only_native_save_load_rejects_wrong_base(self):
        model = TemporalAdapter(copy.deepcopy(self.base))
        state = model.adapter_state()
        self.assertTrue(state and all(not k.startswith("base.") for k in state))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"adapter.pt"
            meta = dict(spec_id="SPEC-109", base_sha256="a"*64, variant="pair", rank=8)
            save_checkpoint(path, state, meta)
            bundle = load_checkpoint(path, sha256_file(path))
            validate_adapter_binding(bundle, "a"*64, "pair")
            with self.assertRaises(ValueError):
                validate_adapter_binding(bundle, "b"*64, "pair")
            restored = TemporalAdapter(copy.deepcopy(self.base))
            restored.load_adapter_state(bundle["state"])
            self.assertTrue(torch.equal(model(self.x), restored(self.x)))
            with self.assertRaises(ValueError):
                restored.load_adapter_state(dict(state, **{"base.head.bias": torch.zeros(2)}))

    def test_empty_wrong_shape_and_nonfinite_are_rejected(self):
        model = TemporalAdapter(self.base)
        for x in (torch.empty(0, 12, 16, 16), torch.zeros(1, 6, 16, 16),
                torch.zeros(1, 12, 17, 16), self.x*float("nan")):
            with self.assertRaises(ValueError):
                model(x)
        with self.assertRaises(ValueError):
            odds_update(torch.tensor([.5]), torch.tensor([float("nan")]))


if __name__ == "__main__":
    unittest.main()
