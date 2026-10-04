import unittest
import numpy as np
from src.comp.reference_additions import additions


class AdditionTest(unittest.TestCase):
    def test_weak_reference_never_removed_and_additions_need_point95(self):
        ref=np.array([[.5001,.2],[.1,.6]]);pair=np.array([[.01,.9499],[.95,.99]])
        result,report=additions(ref,pair,np.ones_like(ref,bool),np.zeros_like(ref,bool))
        self.assertTrue(result[0,0]);self.assertFalse(result[0,1]);self.assertTrue(result[1,0]);self.assertTrue(result[1,1])
        self.assertEqual(report['added_pixels'],1)

    def test_water_invalid_and_nonfinite_fail_closed(self):
        ref=np.ones((2,2))*.6;pair=np.ones_like(ref)
        valid=np.array([[True,False],[True,True]]);water=np.array([[True,False],[False,True]])
        result,_=additions(ref,pair,valid,water)
        self.assertEqual(int(result.sum()),1)
        with self.assertRaises(ValueError):additions(ref,pair*np.nan,valid,water)
        with self.assertRaises(ValueError):additions(ref,pair,valid[:1],water)


if __name__=="__main__":unittest.main()
