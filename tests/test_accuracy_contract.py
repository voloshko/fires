import copy
from pathlib import Path
import tempfile
import unittest

from src.comp.accuracy_contract import (
    bind_cache, event_group, experiment_fingerprint, paired_group_comparison,
    pooled_iou, validate_manifest, validate_pair, verify_cache,
)


def record(sample="a", group="ems:001", partition="fit", bbox=(0, 0, 1, 1)):
    grid = {"crs": "EPSG:32636", "transform": [30, 0, 0, 0, -30, 30],
            "width": 512, "height": 512}
    return {"sample_id": sample, "group_id": group, "partition": partition,
            "label_provenance": "expert_perimeter", "bbox_wgs84": bbox,
            "pre_at": "2025-07-01T00:00:00Z", "start_at": "2025-07-20T00:00:00Z",
            "post_at": "2025-07-25T00:00:00Z", "mgrs": "36SVD", "pre_mgrs": "36SVD",
            "grid": grid, "pre_grid": copy.deepcopy(grid),
            "pre_available_at": "2025-07-01T07:00:00Z",
            "post_available_at": "2025-07-25T07:00:00Z"}


class AccuracyContractTests(unittest.TestCase):
    def test_ems_groups_aois_by_activation(self):
        self.assertEqual(event_group("ems", {"code": "001", "event_id": "001_AOI01"}),
                         event_group("ems", {"code": "001", "event_id": "001_AOI02"}))
        with self.assertRaises(ValueError): event_group("russia", {"id": 1})

    def test_event_leakage(self):
        with self.assertRaisesRegex(ValueError, "event crosses"):
            validate_manifest([record(), record("b", partition="evaluation", bbox=(3, 3, 4, 4))])

    def test_footprint_leakage_and_touching_edges(self):
        with self.assertRaisesRegex(ValueError, "spatial"):
            validate_manifest([record(), record("b", "ems:002", "evaluation", (.5, .5, 2, 2))])
        result = validate_manifest([record(), record("b", "ems:002", "evaluation", (1, 0, 2, 1))])
        self.assertEqual(result["groups"], 2)

    def test_duplicates_are_not_training_weights(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_manifest([record(), record()])

    def test_weak_labels_are_training_only(self):
        r = record(partition="evaluation"); r["label_provenance"] = "weak_candidate"
        with self.assertRaisesRegex(ValueError, "weak labels"): validate_manifest([r])
        r["partition"] = "fit"; validate_manifest([r])

    def test_negative_labels_require_review(self):
        r = record(); r["label_provenance"] = "reviewed_negative"
        with self.assertRaisesRegex(ValueError, "review"): validate_manifest([r])
        r["review_reference"] = "review.json#window-a"; validate_manifest([r])

    def test_pair_tile_grid_and_dates(self):
        validate_manifest([record()], require_pairs=True)
        for key, value in [("pre_mgrs", "35SVD"), ("pre_at", "2025-07-26T00:00:00Z"),
                           ("pre_grid", {"width": 256}), ("start_at", "2025-07-20")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                r = record(); r[key] = value; validate_pair(r)

    def test_asof_requires_acquisition_and_publication(self):
        r = record(); validate_pair(r, "2025-07-25T08:00:00Z")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_pair(r, "2025-07-25T06:00:00Z")
        r["post_available_at"] = "2025-07-24T08:00:00Z"
        with self.assertRaises(ValueError): validate_pair(r, "2025-07-25T08:00:00Z")

    def test_matching_but_invalid_grids_fail(self):
        for key, value in [("width", 0), ("crs", ""), ("transform", [0] * 6)]:
            r = record(); r["grid"][key] = value; r["pre_grid"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_pair(r)

    def test_fingerprint_tracks_order_settings_and_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "input"; p.write_bytes(b"abc")
            records = [record(), record("b", "ems:002", bbox=(2, 0, 3, 1))]
            f = lambda rs=records, cfg="v1": experiment_fingerprint(rs, [p], [p], [p], cfg)
            initial = f(); self.assertEqual(initial, f())
            self.assertNotEqual(initial, f(list(reversed(records))))
            self.assertNotEqual(initial, f(cfg="v2"))
            p.write_bytes(b"abcd"); self.assertNotEqual(initial, f())

    def test_cache_cannot_be_rebound_or_modified(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cache"; p.write_bytes(b"prediction")
            bind_cache(p, "run-a"); verify_cache(p, "run-a")
            with self.assertRaises(ValueError): bind_cache(p, "run-b")
            p.write_bytes(b"other")
            with self.assertRaises(ValueError): verify_cache(p, "run-a")

    def test_bootstrap_clusters_before_resampling(self):
        groups = ["ems:001", "ems:001", "ems:002"]
        a, b = [(8, 1, 1), (6, 2, 2), (7, 1, 2)], [(5, 2, 3), (3, 3, 4), (4, 2, 4)]
        windows = paired_group_comparison(groups, a, b, n_boot=100)
        events = paired_group_comparison(["ems:001", "ems:002"], [(14, 3, 3), a[2]],
                                         [(8, 5, 7), b[2]], n_boot=100)
        self.assertEqual(windows["ci95"], events["ci95"])
        self.assertEqual(windows["groups"], 2)
        self.assertAlmostEqual(windows["candidate_iou"], 21 / 30)
        self.assertEqual(windows["verdict"], "better")

    def test_undefined_and_invalid_counts_fail_closed(self):
        for counts in [(0, 0, 0), (float("nan"), 1, 2), (1, -1, 2), (.5, 1, 2)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError): pooled_iou([counts])
        with self.assertRaises(ValueError): paired_group_comparison(["a"], [(1, 0, 0)], [(1, 0, 0)])


if __name__ == "__main__": unittest.main()
