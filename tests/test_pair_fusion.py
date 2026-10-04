import unittest
import numpy as np
from src.comp.pair_fusion import fuse,select_recipe,confirmation_acceptance


class FusionTest(unittest.TestCase):
    def test_max_preserves_reference_detection_and_mean_does_not(self):
        reference=np.array([[.6,.2],[.8,.1]]);paired=np.array([[.1,.9],[.2,.7]])
        self.assertTrue((fuse(reference,paired,"maximum")[reference>=.5]>=.5).all())
        self.assertLess(fuse(reference,paired,"mean")[0,0],.5)
        with self.assertRaises(ValueError):fuse(reference,paired[:1],"mean")
        with self.assertRaises(ValueError):fuse(reference,paired*np.nan,"mean")
        with self.assertRaises(ValueError):fuse(reference,paired,"optimized")

    def test_mediterranean_losses_can_reject_best_russian_score(self):
        report=lambda iou,missed:dict(russian_vs_c1mm=dict(delta=.02,candidate_iou=iou),
            mediterranean={d:dict(delta=0,candidate_missed_groups=missed,baseline_missed_groups=0) for d in ("ems","floga")})
        reports=dict(mean=report(.4,1),maximum=report(.35,0))
        self.assertEqual(select_recipe(reports),"maximum")
        reports["maximum"]["mediterranean"]["floga"]["delta"]=-.02
        self.assertIsNone(select_recipe(reports))

    def test_confirmation_requires_both_named_wins(self):
        good=dict(delta=.03,ci95=[.01,.05]);ambiguous=dict(delta=.02,ci95=[-.01,.05])
        c=dict(c1mm=good,prithvi=good,reserve=dict(delta=0))
        self.assertTrue(confirmation_acceptance(c,dict(fusion=0,c1mm=0)))
        c["c1mm"]=ambiguous
        self.assertFalse(confirmation_acceptance(c,dict(fusion=0,c1mm=0)))


if __name__=="__main__":unittest.main()
