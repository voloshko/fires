# SPEC-100: Paired severity confirmation within PC coverage

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-94, SPEC-95, SPEC-99

## Summary

SPEC98 has too few unused2024events; SPEC99 predates PC HLS coverage. Metadata
only audit2020–2024 finds240 independent candidates (no model predictions).
Confirm frozen equal SPEC94 severity against frozenhybrid on previously unused
CONUS events within actual source coverage. Retrospective, no prospective claim.

## Acceptance Criteria

- Explicit profile pc-covered-2020-24, official MTBS2020–2024 CONUS, extent<=14km,
  excludeAK/HI/PR; SHA256('spec100:'+event_id), cap240 independent512x51230m UTM
  centroid frames. All prior2658frame intersections, oldfresh1–4 events and
  intersections among selectedframes excluded before predictions, asSPEC98.
  Freeze selection and source temporal metadata before imagery; keep failures.
- Reuse strict SPEC98/99 collection: PC HLS post30–150days, prewithin90days
  strictly before ignition, sameMGRS/grid, commonvalid, <10%cloud/nodata eachdate;
  MTBS export for exact eventyear with nearestgrid, source-query/hash sidecar,
  knownclasses0–6 checked even on cachedrecords. Water Fmask alsohash-bound;
  cachedseverity counts recomputed from raster/commonvalid. Minimum8GiB free,
  workers2, ownk8plus outputs only; no old artifacts overwritten or deleted.
- At least100 admitted distinct freshevent groups, all3grades represented,
  or INSUFFICIENT_EVIDENCE; no selection changes after predictions.
- Frozen SPEC94 result SHA
  5f6b2839b89c807692dc4c931b2f4e713ae03294f869cd5233faf89a5617999e;
  exact equal-s1/2/3 modelhashes/manifest/recipe and5boundaryhashes. No training,
  normalization from checkpoints, no changing thresholds/recipe on thisset.
- Same fixedC1-MM boundary0.5/TTA8/BF16 andwatermask, labels asSPEC94.
  Primary equal−hybrid mIoU>=.02, event-bootstrapCI95lower>0, no perclassdrop>.02.
  Bootstrap2000 seed20261003. Persist counts/source/model/input fingerprints.
- Source/profile/cache regressions and makegate pass. GPUscore serialized
  after Russian queue and fulladmission under research/accuracy-gpu.lock;
  recipe rechecked after waiting. Never run ML/downloads onomarchy/Mac.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_fresh_paired_severity.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/prepare_fresh_paired_severity.py --profile pc-covered-2020-24 --data-root data-store --source-root /home/mc/fires --prior data-store/prior-footprints.json --out data-store/severity-covered-v1`.
- k8plus: `bash scripts/queues/queue_covered_paired_severity.sh`.
- non_claims: published satellite MTBS agreement only, retrospective2020–2024,
  no fieldtruth, no Siberian severity proof, no globalrank or victory over
  untested competitors. QC/model-load tests cannot prove real accuracy.
