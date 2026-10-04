import unittest
import numpy as np
from src.comp.component_rescue import rescue


class RescueTest(unittest.TestCase):
    def test_diagonal_small_component_restored_as_whole(self):
        ref=np.zeros((3,3));ref[0,0]=ref[1,1]=.6
        result,report=rescue(ref,np.zeros_like(ref),np.ones_like(ref,bool),np.zeros_like(ref,bool))
        self.assertEqual(int(result.sum()),2);self.assertEqual(report["restored_components"],1)

    def test_large_and_already_retained_components_not_rescued(self):
        ref=np.ones((9,9))*.6;pair=np.zeros_like(ref)
        result,report=rescue(ref,pair,np.ones_like(ref,bool),np.zeros_like(ref,bool))
        self.assertEqual(int(result.sum()),0);self.assertEqual(report["restored_pixels"],0)
        ref=np.zeros((3,3));ref[0,:2]=.6;pair=np.zeros_like(ref);pair[0,0]=.9
        result,report=rescue(ref,pair,np.ones_like(ref,bool),np.zeros_like(ref,bool))
        self.assertEqual(int(result.sum()),1);self.assertEqual(report["restored_pixels"],0)

    def test_water_invalid_and_wrong_grid_never_restored(self):
        ref=np.ones((2,2))*.6;pair=np.zeros_like(ref)
        valid=np.array([[True,False],[False,False]]);water=np.ones_like(valid)
        result,_=rescue(ref,pair,valid,water);self.assertFalse(result.any())
        with self.assertRaises(ValueError):rescue(ref,pair,valid[:1],water)


if __name__=="__main__":unittest.main()
