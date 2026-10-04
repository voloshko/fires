import unittest
import numpy as np

from src.comp.temporal_adapter_evaluation import pack_prediction, unpack_prediction, development_decision


class AdapterEvaluationTest(unittest.TestCase):
    def test_packed_masks_are_lossless_at_threshold_boundary(self):
        probability = np.array([0., .499999, .5, .500001, 1., .1, .9, .5], np.float32)[None, None]
        mask = probability >= .5
        self.assertTrue(np.array_equal(unpack_prediction(pack_prediction(mask), mask.shape), mask))
        with self.assertRaises(ValueError):
            pack_prediction(probability)

    def datasets(self):
        return {name: [dict(sample_id=name+str(i), group_id=name+str(i), valid_pixels=30, total_pixels=32,
            pair=[9, 1, 1], post=[6, 2, 4], reserve=[8, 1, 2], c1mm=[5, 2, 5], prithvi=[4, 3, 6])
            for i in range(2)] for name in ("russia_eval", "russia_cal", "western", "ems", "floga")}

    def test_all_competitors_and_weak_group_recall_are_required(self):
        datasets = self.datasets()
        self.assertTrue(development_decision(datasets)["eligible"])
        datasets["russia_cal"][0]["pair"] = [0, 1, 10]
        result = development_decision(datasets)
        self.assertFalse(result["eligible"])
        self.assertFalse(result["guards"]["missed_groups"])

    def test_missing_population_or_truth_mismatch_is_not_a_zero(self):
        datasets = self.datasets()
        datasets.pop("floga")
        with self.assertRaises(ValueError):
            development_decision(datasets)
        datasets = self.datasets()
        datasets["western"][0]["post"] = [6, 2, 5]
        with self.assertRaises(ValueError):
            development_decision(datasets)


if __name__ == "__main__":
    unittest.main()
