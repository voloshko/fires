import unittest
import numpy as np
from src.comp.mtbs_legend_audit import attribute


class MTBSLegendTest(unittest.TestCase):
    def test_raw_code_decomposition_recombines_and_does_not_relabel(self):
        raw = np.array([[0, 1, 2, 3, 4, 5, 6, 255]])
        prediction = np.array([[1, 0, 1, 2, 3, 1, 0, 0]])
        report = attribute(raw, prediction, np.ones_like(raw, bool))
        cm = sum(np.array(v["legacy_confusion"]) for v in report["by_raw_code"].values())
        self.assertTrue(np.array_equal(cm, report["legacy_confusion"]))
        self.assertEqual(report["by_raw_code"]["0"]["legacy_errors"], 1)
        self.assertEqual(report["by_raw_code"]["5"]["legacy_errors"], 1)
        self.assertIsNone(report["by_raw_code"]["6"]["legacy_errors"])
        self.assertEqual(report["unknown_codes"], [255])

    def test_invalid_observation_never_becomes_known_background(self):
        report = attribute(np.array([[0, 2]]), np.array([[3, 1]]), np.array([[False, True]]))
        self.assertNotIn("0", report["by_raw_code"])
        self.assertEqual(report["observed_pixels"], 1)

    def test_bad_geometry_and_impossible_prediction_rejected(self):
        with self.assertRaises(ValueError):
            attribute(np.zeros((2, 2)), np.zeros((1, 2)), np.ones((2, 2), bool))
        with self.assertRaises(ValueError):
            attribute(np.zeros((2, 2)), np.full((2, 2), 4), np.ones((2, 2), bool))


if __name__ == "__main__":
    unittest.main()
