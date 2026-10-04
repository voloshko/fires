import unittest
import numpy as np
import torch

from src.comp.temporal_adapter_contract import exposure_pool, fit_population, tensor_digest


class AdapterContractTest(unittest.TestCase):
    def test_all_fit_views_survive_repetition_and_partial_batch(self):
        rows = [dict(sample_id=str(i), partition="fit", sampling_weight=2 if i < 79 else 1) for i in range(683)]
        order = np.random.default_rng(1).permutation(exposure_pool(rows)).tolist()
        batches = [order[i:i+8] for i in range(0, len(order), 8)]
        self.assertEqual(len(order), 762)
        self.assertEqual(len(batches[-1]), 2)
        self.assertEqual(set(i for b in batches for i in b), set(range(683)))
        self.assertTrue(all(order.count(i) == rows[i]["sampling_weight"] for i in range(683)))
        rows[0]["partition"] = "evaluation"
        with self.assertRaises(ValueError):
            exposure_pool(rows)

    def test_incomplete_or_duplicate_frozen_population_rejected(self):
        base = [dict(sample_id=f"b{i}", partition="fit") for i in range(374)]
        russian = [dict(sample_id=f"r{i}", partition="fit") for i in range(309)]
        self.assertEqual(len(fit_population(base, russian)), 683)
        with self.assertRaises(ValueError):
            fit_population(base, russian[:-1])
        russian[0]["sample_id"] = "b0"
        with self.assertRaises(ValueError):
            fit_population(base, russian)

    def test_frozen_tensor_digest_covers_scalar_buffers_and_values(self):
        state = dict(weight=torch.arange(6.).reshape(2, 3), counter=torch.tensor(0))
        digest = tensor_digest(state)
        self.assertEqual(digest, tensor_digest(state))
        state["counter"] += 1
        self.assertNotEqual(digest, tensor_digest(state))


if __name__ == "__main__":
    unittest.main()
