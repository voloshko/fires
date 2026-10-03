"""SPEC-98: evidence admission must fail before a stale/contaminated comparison."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from src.comp.fresh_severity import isolate, assert_fresh, verify_frozen_models, verify_development_result, population


class FreshSeverityTest(unittest.TestCase):
    def test_populations_and_source_year_are_explicit(self):
        from scripts.prepare_fresh_paired_severity import severity_identity
        import rasterio
        self.assertEqual(population("retrospective-2016-17"),("SPEC-99",{2016,2017},"spec99:"))
        self.assertEqual(population("2024-west")[0],"SPEC-98")
        with self.assertRaises(ValueError):population("adaptive")
        grid=dict(crs="EPSG:32611",transform=rasterio.Affine(30,0,300000,0,-30,4000000))
        self.assertIn("mtbs_CONUS_2016",severity_identity(grid,2016)["query"]["mosaicRule"])
        with self.assertRaises(ValueError):severity_identity(grid,2023)

    def test_modified_development_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"result.json";path.write_text('{"model_hashes":[]}')
            with self.assertRaises(ValueError):verify_development_result(path)

    def test_cached_severity_requires_grid_classes_and_source(self):
        import json
        import numpy as np
        import rasterio
        from scripts.prepare_fresh_paired_severity import validate_severity, validate_cached_record, severity_identity, SEVERITY_URL
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"severity.tif"
            grid=dict(crs="EPSG:32611",transform=rasterio.Affine(30,0,300000,0,-30,4000000))
            identity=dict(url="official export",query=dict(year=2024))
            def write(width=512,label=2):
                with rasterio.open(path,"w",driver="GTiff",width=width,height=512,count=1,dtype="uint8",**grid) as target:
                    target.write(np.full((1,512,width),label,np.uint8))
                path.with_suffix(".source.json").write_text(json.dumps(dict(identity=identity,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
            write();self.assertEqual(validate_severity(path,grid,identity).shape,(512,512))
            for width,label in ((256,2),(512,9)):
                write(width,label)
                with self.assertRaises(ValueError):validate_severity(path,grid,identity)
            write()
            with self.assertRaises(ValueError):validate_severity(path,dict(grid,transform=rasterio.Affine(30,0,300030,0,-30,4000000)),identity)
            with self.assertRaises(ValueError):validate_severity(path,grid,dict(identity,url="different source"))
            path.with_suffix(".source.json").unlink()
            with self.assertRaises(ValueError):validate_severity(path,grid,identity)
            event=dict(event_id="new",grid=dict(grid,transform=list(grid["transform"])[:6],width=512,height=512))
            record=dict(group_id="mtbs:new",grid=event["grid"],partition="evaluation",severity_source=SEVERITY_URL,
                        files=dict(severity="severity.tif"),hashes=dict(severity=hashlib.sha256(path.read_bytes()).hexdigest()))
            with self.assertRaises(ValueError):validate_cached_record(Path(directory),record,event)
            path.with_suffix(".source.json").write_text(json.dumps(dict(identity=severity_identity(grid),sha256=record["hashes"]["severity"])))
            validate_cached_record(Path(directory),record,event)

    def test_excludes_old_events_and_frames_before_cap(self):
        rows=[dict(event_id=e,bbox_wgs84=b) for e,b in (
            ("old",[0,0,1,1]),("prior-overlap",[2,2,4,4]),
            ("new",[5,5,6,6]),("new-overlap",[5.5,5.5,7,7]),("next",[8,8,9,9]))]
        keep,excluded=isolate(rows,{"old"},[dict(bbox_wgs84=[3,3,5,5])],limit=2)
        self.assertEqual([r["event_id"] for r in keep],["new","next"])
        self.assertEqual(len(excluded),3)

    def test_detects_model_replacement_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/str(i) for i in range(2)]
            for i,p in enumerate(paths):p.write_bytes(str(i).encode())
            hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
            verify_frozen_models(paths,hashes)
            with self.assertRaises(ValueError):verify_frozen_models(paths,hashes[::-1])
            paths[0].write_bytes(b"different model")
            with self.assertRaises(ValueError):verify_frozen_models(paths,hashes)

    def test_cannot_substitute_selected_frame_or_train(self):
        grid=dict(crs="EPSG:32611",transform=[30,0,0,0,-30,0],width=512,height=512)
        plan=dict(used_events=["old"],selected=[dict(event_id="new",grid=grid)])
        record=dict(group_id="mtbs:new",partition="evaluation",grid=grid)
        assert_fresh([record],plan)
        for change in (dict(group_id="mtbs:old"),dict(partition="fit"),dict(grid=dict(grid,width=256))):
            with self.assertRaises(ValueError):assert_fresh([dict(record,**change)],plan)
        with self.assertRaises(ValueError):assert_fresh([record,record],plan)


if __name__=="__main__":unittest.main()
