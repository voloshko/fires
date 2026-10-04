import unittest
from src.comp.western_holdout import exclude_groups,balanced_selection,storage_guard,BUDGET_BYTES,FREE_FLOOR_BYTES,verify_exclusions,EXCLUSION_SOURCES


class WesternTest(unittest.TestCase):
    def test_each_incomplete_exclusion_source_rejected_before_selection(self):
        for name,(count,spec,digest) in EXCLUSION_SOURCES.items():
            rows=[dict(sample_id=str(i),bbox_wgs84=[65,54,66,55],source_sha256="a"*64,hashes=dict(pre="a"*64)) for i in range(count)]
            payload=dict(spec_id=spec,records=rows,ready_for_training=True,ready_for_evaluation=True,qc_probe=False)
            self.assertEqual(verify_exclusions(name,payload,digest)['records'],count)
            with self.assertRaises(ValueError):verify_exclusions(name,dict(payload,records=rows[:-1]),digest)
            with self.assertRaises(ValueError):verify_exclusions(name,payload,"b"*64)

    def test_one_overlap_removes_entire_group(self):
        rows=[dict(sample_id=str(i),group_id=g) for i,g in enumerate(("a","a","b"))]
        keep,rejected=exclude_groups(rows,{"0"})
        self.assertEqual([r["sample_id"] for r in keep],["2"]);self.assertEqual(rejected,{"a"})

    def test_groups_distributed_before_more_windows(self):
        rows=[dict(sample_id=f"{g}-{i}",group_id=g,area_ha=a) for g in ("a","b","c") for i,a in enumerate((1,30,300,2,40))]
        chosen=balanced_selection(rows,cap=3)
        self.assertEqual(len({r["group_id"] for r in chosen}),3)
        all_=balanced_selection(rows)
        self.assertEqual(len(all_),12);self.assertTrue(all(r["area_ha"]>=1 for r in all_))

    def test_budget_accounts_concurrent_reservations_and_free_floor(self):
        storage_guard(100,100,FREE_FLOOR_BYTES+100)
        with self.assertRaises(ValueError):storage_guard(BUDGET_BYTES,1,FREE_FLOOR_BYTES+10)
        with self.assertRaises(ValueError):storage_guard(100,100,FREE_FLOOR_BYTES+99)


if __name__=="__main__":unittest.main()
