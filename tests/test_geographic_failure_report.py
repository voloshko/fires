import unittest

import numpy as np

from src.comp.accuracy_contract import canonical_hash
from src.comp.geographic_failure_report import (
    diagnostic_masks, group_influence, spectral_indices, spectral_summary, verify_recipe, verify_rows,
)


class GeographicFailureTest(unittest.TestCase):
    def test_recipe_body_cannot_change_behind_unchanged_id(self):
        recipe = dict(source_hashes={"score": "a"}, configuration={"threshold": .5})
        expected = canonical_hash(recipe)
        recipe["recipe_id"] = expected
        verify_recipe(recipe, expected)
        recipe["configuration"]["threshold"] = .4
        with self.assertRaises(ValueError):
            verify_recipe(recipe, expected)

    def rows(self):
        return [dict(sample_id=str(i), group_id=g, valid_pixels=20, total_pixels=20,
            reserve=a, c1mm=b, prithvi=b) for i, (g, a, b) in enumerate([
                ("a", [2, 1, 2], [3, 2, 1]), ("a", [1, 0, 3], [2, 1, 2]),
                ("b", [3, 1, 1], [1, 1, 3])])]

    def test_wrong_identity_group_or_counts_abort(self):
        rows = self.rows()
        reconstructed = [{k: v for k, v in r.items() if k != "total_pixels"} for r in rows]
        verify_rows(rows, rows, reconstructed)
        for bad in (list(reversed(rows)), [dict(rows[0], group_id="x")]+rows[1:]):
            with self.assertRaises(ValueError):
                verify_rows(bad, rows, reconstructed)
        changed = [dict(reconstructed[0], reserve=[3, 1, 1])]+reconstructed[1:]
        with self.assertRaises(ValueError):
            verify_rows(rows, rows, changed)

    def test_group_influence_removes_all_windows(self):
        groups = group_influence(self.rows())
        a = next(r for r in groups if r["group_id"] == "a")
        self.assertEqual(a["windows"], 2)
        self.assertEqual(a["counts"]["reserve"], [3, 1, 5])
        self.assertAlmostEqual(a["delta_without_group"], 3/5-1/5)

    def test_zero_denominator_and_empty_subsets_remain_explicit(self):
        pre = np.zeros((6, 1, 2), np.float32)
        post = np.ones((6, 1, 2), np.float32)
        indices = spectral_indices(pre, post)
        report = spectral_summary(indices, np.ones((1, 2), bool))
        self.assertEqual(report["indices"]["pre_nbr"]["undefined_pixels"], 2)
        self.assertIsNone(report["indices"]["pre_nbr"]["q10_median_q90"])
        empty = spectral_summary(indices, np.zeros((1, 2), bool))
        self.assertEqual(empty["pixels"], 0)
        self.assertIsNone(empty["indices"]["post_nbr"]["q10_median_q90"])

    def test_invalid_water_and_ignored_labels_never_become_fp(self):
        masks = diagnostic_masks(np.array([[0, 0, 0, -1, 1]]),
            np.array([[True, False, True, True, True]]), np.array([[False, False, True, False, False]]),
            np.ones((1, 5)), np.zeros((1, 5)))
        self.assertEqual(masks["reserve_only_fp"].tolist(), [[True, False, False, False, False]])
        self.assertEqual(masks["labelled_burn"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
