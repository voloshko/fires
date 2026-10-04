"""SPEC-105: storage and whole-group exclusion, independent of predictions."""
from collections import defaultdict,deque
from pathlib import Path
import shutil
import math
from threading import Lock
from .accuracy_contract import canonical_hash

BUDGET_BYTES=1536*1024**2
FREE_FLOOR_BYTES=3*1024**3
FRAME_RESERVATION_BYTES=14*1024**2

EXCLUSION_SOURCES={
    "prior-footprints.json":(2658,"SPEC-96","eb7bb1af7800c00e1837cca0634ffcad82792bf1dcd41cbd06fe112ea9aeecad"),
    "russia/manifest.json":(447,"SPEC-92","1d487c711beb657a4c0ea0114379ddc51be2a31302891e47699942f95459f3db"),
    "severity-covered-v1/manifest.json":(173,"SPEC-100","ed7903750216a4e341a394908dbe06f44fd195e0ade8f67034ec8497a83e79dc"),
}


def verify_exclusions(name,payload,digest):
    count,spec,expected=EXCLUSION_SOURCES[name]
    records=payload.get("records",[])
    if digest!=expected or payload.get("spec_id")!=spec or len(records)!=count:
        raise ValueError("exclusion source identity/population differs")
    if name=="russia/manifest.json" and (not payload.get("ready_for_training") or payload.get("qc_probe")):
        raise ValueError("Russian exclusion source provisional")
    if name=="severity-covered-v1/manifest.json" and (not payload.get("ready_for_evaluation") or payload.get("qc_probe")):
        raise ValueError("fresh exclusion source provisional")
    ids=set()
    for record in records:
        if not record.get("sample_id") or record["sample_id"] in ids:raise ValueError("exclusion identities missing/duplicate")
        ids.add(record["sample_id"])
        bounds=record.get("bbox_wgs84",[])
        if (len(bounds)!=4 or not all(math.isfinite(v) for v in bounds)
                or not -180<=bounds[0]<bounds[2]<=180 or not -90<=bounds[1]<bounds[3]<=90):
            raise ValueError("exclusion footprint invalid")
        hashes=[record.get("source_sha256")] if name=="prior-footprints.json" else list(record.get("hashes",{}).values())
        if not hashes or any(not isinstance(h,str) or len(h)!=64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("exclusion record missing input hashes")
    return dict(spec_id=spec,records=count,sha256=digest)


def exclude_groups(records,bad_ids):
    rejected={r["group_id"] for r in records if r["sample_id"] in bad_ids}
    return [r for r in records if r["group_id"] not in rejected],rejected


def balanced_selection(records,cap=96):
    groups=defaultdict(list)
    for r in records:
        if r["area_ha"]>=1:groups[r["group_id"]].append(r)
    choices={}
    for group,rows in groups.items():
        buckets=[deque(sorted([r for r in rows if (0 if r["area_ha"]<25 else 1 if r["area_ha"]<250 else 2)==i],
                              key=lambda r:canonical_hash("spec105:"+r["sample_id"]))) for i in range(3)]
        selected=[]
        while any(buckets) and len(selected)<4:
            for bucket in buckets:
                if bucket and len(selected)<4:selected.append(bucket.popleft())
        choices[group]=selected
    order=sorted(choices,key=lambda g:canonical_hash("spec105:"+g))
    return [choices[g][round_] for round_ in range(4) for g in order if round_<len(choices[g])][:cap]


def storage_guard(used,reserved,free):
    if used<0 or reserved<0 or free<0:raise ValueError("invalid storage accounting")
    if used+reserved>BUDGET_BYTES:raise ValueError("new data budget exhausted")
    if free-reserved<FREE_FLOOR_BYTES:raise ValueError("free-space reserve unavailable")


class StorageBudget:
    def __init__(self,directory):
        self.directory=Path(directory);self.lock=Lock();self.reserved=0
    def used(self):
        files=list(self.directory.rglob("*"))
        if any(p.is_symlink() for p in files):raise ValueError("output storage contains symlink")
        return sum(p.stat().st_size for p in files if p.is_file())
    def reserve(self):
        with self.lock:
            storage_guard(self.used(),self.reserved+FRAME_RESERVATION_BYTES,shutil.disk_usage(self.directory).free)
            self.reserved+=FRAME_RESERVATION_BYTES
    def release(self):
        with self.lock:self.reserved-=FRAME_RESERVATION_BYTES
