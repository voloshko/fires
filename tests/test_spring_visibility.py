import unittest

import numpy as np

from src.comp.spring_visibility import (BUDGET_BYTES, FREE_FLOOR_BYTES, RESERVATION_BYTES,
    candidates, compare_view, decision, storage_guard)


class SpringVisibilityTest(unittest.TestCase):
    def test_metadata_date_tile_cloud_and_order(self):
        good = dict(item_id="b", date="2020-04-02T10:00:00Z", mgrs="tile", cloud=2)
        rows = [good, dict(good, item_id="a"), dict(good, item_id="other", mgrs="other"),
            dict(good, item_id="late", date="2020-05-02T10:00:00Z"),
            dict(good, item_id="unknown", cloud=None), dict(good, item_id="nan", cloud=float("nan"))]
        self.assertEqual([r["item_id"] for r in candidates(rows, "tile", "2020-04-01", "2020-04-30")], ["a", "b"])

    def test_common_mask_prevents_false_spectral_improvement(self):
        image = np.ones((6, 10, 8), np.float32)
        labels = np.zeros((10, 8), np.int16)
        labels[:, :4] = 1
        image[3, :, :3] = 3  # Most old GT pixels have a different NBR.
        valid = np.ones((10, 8), bool)
        current_valid = valid.copy()
        current_valid[:, :3] = False
        report = compare_view(image, image.copy(), labels, valid, current_valid)
        self.assertTrue(report["usable"])
        self.assertEqual(report["contrast"]["old"], report["contrast"]["current"])
        self.assertFalse(report["extra_separation"])

    def test_absence_is_insufficient_not_negative(self):
        ids = [str(i) for i in range(4)]
        self.assertEqual(decision([], ids)["status"], "INSUFFICIENT_EVIDENCE")
        rows = [dict(sample_id=i, usable=True, extra_separation=True, date=d)
            for i in ids[:2] for d in ("2020-04-01T00:00:00Z", "2020-04-15T00:00:00Z")]
        self.assertTrue(decision(rows, ids)["supports_multidate_followup"])
        for r in rows:
            r["extra_separation"] = False
        self.assertEqual(decision(rows, ids)["status"], "FAIL")

    def test_budget_and_disk_floor_include_reservation(self):
        storage_guard(0, FREE_FLOOR_BYTES+RESERVATION_BYTES)
        with self.assertRaises(ValueError):
            storage_guard(BUDGET_BYTES-RESERVATION_BYTES+1, FREE_FLOOR_BYTES+RESERVATION_BYTES)
        with self.assertRaises(ValueError):
            storage_guard(0, FREE_FLOOR_BYTES+RESERVATION_BYTES-1)


if __name__ == "__main__":
    unittest.main()
