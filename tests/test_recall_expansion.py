import copy
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.train_unet import UNet
from src.comp.accuracy_artifacts import load_checkpoint, save_checkpoint
from src.comp.accuracy_contract import sha256_file
from src.comp.recall_expansion import (RecallExpansion, expand_probability,
    expansion_log_probs, validate_expansion_binding)


class RecallExpansionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(2)
        self.base = UNet(6, classes=2, w=4, depth=3).eval()
        self.x = torch.randn(2, 12, 16, 16)

    def test_monotonic_expansion_keeps_weak_detections_and_endpoints(self):
        p = torch.tensor([0., .2, .499999, .5, .58, .99, 1.])
        for q in (torch.zeros_like(p), torch.full_like(p, .5), torch.ones_like(p)):
            output = expand_probability(p, q)
            self.assertTrue((output >= p).all())
            self.assertTrue(((p >= .5) <= (output >= .5)).all())
            self.assertEqual(float(output[-1]), 1)
        self.assertGreater(float(expand_probability(p, torch.full_like(p, .9))[0]), .5)

    def test_log_probability_matches_formula_and_is_stable_with_gradient(self):
        original = torch.randn(2, 2, 4, 4)*20
        value = torch.full((2, 4, 4), -4., requires_grad=True)
        logs = expansion_log_probs(original, value)
        expected = expand_probability(original.softmax(1)[:, 1], value.sigmoid())
        self.assertTrue(torch.allclose(logs.exp().sum(1), torch.ones_like(value), atol=1e-6))
        self.assertTrue(torch.allclose(logs.exp()[:, 1], expected, atol=1e-6))
        (-logs[:, 1].mean()).backward()
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_backbone_bn_frozen_and_declared_prior_has_gradient(self):
        model = RecallExpansion(copy.deepcopy(self.base)).train()
        state = {k: v.clone() for k, v in model.base.state_dict().items()}
        logs, q = model.forward_with_addition(self.x)
        self.assertTrue(torch.allclose(q, torch.full_like(q, torch.sigmoid(torch.tensor(-4.)).item())))
        (-logs[:, 1].mean()).backward()
        self.assertGreater(float(model.correction_head[-1].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertTrue(all(torch.equal(v, model.base.state_dict()[k]) for k, v in state.items()))

    def test_pair_uses_pre_control_ignores_it(self):
        changed = self.x.clone(); changed[:, :6] += 5
        for variant in ('pair', 'post'):
            model = RecallExpansion(copy.deepcopy(self.base), variant).eval()
            torch.nn.init.constant_(model.correction_head[-1].weight, .01)
            a = model.forward_with_addition(self.x)[1]
            b = model.forward_with_addition(changed)[1]
            self.assertEqual(torch.equal(a, b), variant == 'post')

    def test_native_state_binding_and_bad_probability_rejected(self):
        model = RecallExpansion(copy.deepcopy(self.base))
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'expansion.pt'
            save_checkpoint(p, model.adapter_state(), dict(spec_id='SPEC-111', base_sha256='a'*64, variant='pair', rank=8))
            bundle = load_checkpoint(p, sha256_file(p)); validate_expansion_binding(bundle, 'a'*64, 'pair')
            restored = RecallExpansion(copy.deepcopy(self.base)); restored.load_adapter_state(bundle['state'])
            self.assertTrue(torch.equal(model(self.x), restored(self.x)))
            with self.assertRaises(ValueError):validate_expansion_binding(bundle, 'b'*64, 'pair')
        for q in (torch.tensor([float('nan')]), torch.tensor([-1.])):
            with self.assertRaises(ValueError):expand_probability(torch.tensor([.58]), q)


if __name__ == '__main__':
    unittest.main()
