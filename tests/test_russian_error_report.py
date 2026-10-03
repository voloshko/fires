import unittest
from src.comp.russian_error_report import summarize,diagnostic_subset


class ErrorReportTest(unittest.TestCase):
    def test_missed_windows_do_not_become_missed_clusters(self):
        rows=[dict(sample_id="a",group_id="one",valid_pixels=10,total_pixels=20,c1mm=[0,2,3],reserve=[1,1,2]),
              dict(sample_id="b",group_id="one",valid_pixels=10,total_pixels=20,c1mm=[2,1,1],reserve=[2,1,1])]
        report=summarize(rows,["c1mm","reserve"])
        self.assertEqual(report["models"]["c1mm"]["missed_positive_windows"],["a"])
        self.assertEqual(report["models"]["c1mm"]["missed_positive_groups"],[])
        self.assertEqual(report["masked_fraction"],.5)
        self.assertAlmostEqual(report["models"]["c1mm"]["fp_fraction_of_known_background"],3/14)

    def test_cross_model_observation_mismatch_is_rejected(self):
        row=dict(sample_id="a",group_id="one",valid_pixels=10,total_pixels=20,c1mm=[0,2,3],reserve=[1,1,1])
        with self.assertRaises(ValueError):summarize([row],["c1mm","reserve"])
        row["reserve"]=[1,10,2]
        with self.assertRaises(ValueError):summarize([row],["c1mm","reserve"])
        row["reserve"]=[1,float("nan"),2]
        with self.assertRaises(ValueError):summarize([row],["c1mm","reserve"])

    def test_hard_subset_uses_reference_only_and_stable_ties(self):
        rows=[dict(sample_id=str(i),c1mm=[1,5,2],reserve=[1,99-i,2]) for i in range(8)]
        self.assertEqual([r["sample_id"] for r in diagnostic_subset(rows[::-1])],["0","1"])


if __name__=="__main__":unittest.main()
