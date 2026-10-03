import unittest
import hashlib
from pathlib import Path
import tempfile
import numpy as np
from src.comp.l2a_full_replay import epoch_view,weighted_order,original_valid,matched_view


class ReplayTest(unittest.TestCase):
    def test_seed_four_or_five_replacement_is_rejected(self):
        from scripts.l2a_full_replay_io import verify_baseline
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/str(i) for i in range(5)]
            for i,path in enumerate(paths):path.write_bytes(str(i).encode())
            hashes=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            verify_baseline(paths,hashes)
            for index in (3,4):
                paths[index].write_bytes(b"different weights")
                with self.assertRaises(ValueError):verify_baseline(paths,hashes)
                paths[index].write_bytes(str(index).encode())

    def test_original_identity_substitution_and_stale_pixels_rejected(self):
        from scripts.l2a_full_replay_io import verify_inputs,verify_original_population
        job=dict(sample_id="a",source="original.tif",partition="fit",dataset="hls",group_id="hls:T00",sampling_weight=1)
        record=dict(job,files=dict(hls="original.tif"))
        verify_original_population([record],[job])
        with self.assertRaises(ValueError):verify_original_population([dict(record,sample_id="other")],[job])
        with self.assertRaises(ValueError):verify_original_population([dict(record,files=dict(hls="other.tif"))],[job])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/"original.tif";path.write_bytes(b"original pixels")
            record["hashes"]=dict(hls=hashlib.sha256(path.read_bytes()).hexdigest())
            verify_inputs(root,record)
            path.write_bytes(b"substituted pixels")
            with self.assertRaises(ValueError):verify_inputs(root,record)

    def test_full_supervision_every_odd_epoch_even_with_missing_twin(self):
        for twin in (False,True):
            for variant in ("control","replay"):
                views=[epoch_view(e,twin,variant) for e in range(1,31)]
                self.assertEqual(sum(mask=="original" for _,mask in views),15 if twin else 30)
        self.assertEqual(epoch_view(2,True,"control"),("hls","common"))
        self.assertEqual(epoch_view(2,True,"replay"),("l2a","common"))

    def test_every_record_including_last_partial_batch_is_preserved(self):
        records=[dict(sampling_weight=w) for w in (1,2,1,1)]
        one=weighted_order(records,np.random.default_rng(1));two=weighted_order(records,np.random.default_rng(1))
        np.testing.assert_array_equal(one,two)
        self.assertEqual(sorted(one.tolist()),[0,1,1,2,3])
        batches=[one[p:p+2] for p in range(0,len(one),2)]
        self.assertEqual([len(b) for b in batches],[2,2,1])

    def test_same_mask_fill_and_full_original_gt_return(self):
        hls=np.ones((6,2,2),np.float32);l2a=hls*2;labels=np.array([[1,0],[1,-1]],np.int8)
        original=original_valid(hls,labels);common=np.array([[True,True],[False,False]])
        hx,hy,hv=matched_view(hls,labels,original,common)
        lx,ly,lv=matched_view(l2a,labels,original,common)
        np.testing.assert_array_equal(hy,ly);np.testing.assert_array_equal(hv,lv)
        self.assertTrue((hx[:,~hv]==0).all() and (lx[:,~lv]==0).all())
        _,full,_=matched_view(hls,labels,original)
        self.assertEqual(full[1,0],1)
        l2a[:,0,0]=-9999
        with self.assertRaises(ValueError):matched_view(l2a,labels,original,common)


if __name__=="__main__":unittest.main()
