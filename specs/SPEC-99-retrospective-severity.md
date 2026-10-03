# SPEC-99: Retrospective independent paired severity confirmation

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-94, SPEC-95, SPEC-98

## Summary

SPEC98 could not admit100 freshwestern2024 events (only21; allCONUS only57).
Metadata-only audit finds at least180 spatially independent2016–2017 CONUS
candidates. Reuse the frozen SPEC94 equal checkpoints, choose this population
before predictions, and confirm the paired severity signal retrospectively.
Do not describe this as prospective or2024 evidence. No new training.

## Acceptance Criteria

- Same SPEC98 collection, isolation, labels, source/model/hash checks and
  untouched old data. New explicit collector profile retrospective-2016-17:
  official MTBS2016/2017 CONUS (exclude AK/HI/PR), event extent<=14km,
  SHA256('spec99:'+event_id) order, max180 nonoverlapping512x51230m UTM frames.
  Exclude every prior2658frame and oldfresh1–4 event. Freeze selection before
  image collection, retain every exclusion and QC failure, no replacement
  based on model prediction. No mixing with SPEC98 inputs/manifests.
- Source-specific MTBS year from verified event date (2016 or2017), exact export
  query/CRS/grid/classes0–6 and hash recorded and checked even for cached files.
  PC hls2-s30 paired dates satisfy pre<ignition<=post, commonvalid pixels,
  sameMGRS/grid; original cloud masks remain. Free disk>=8GiB, workers2.
- Full evaluation requires>=100 distinct admitted freshevents, all3 grade
  classes represented; otherwise INSUFFICIENT_EVIDENCE with no model conclusion.
- Frozen SPEC94 result SHA256
  5f6b2839b89c807692dc4c931b2f4e713ae03294f869cd5233faf89a5617999e;
  exact3 equal checkpoint hashes/recipe/manifest and5 C1-MM boundary hashes.
  Fit statistics from checkpoint only; no retraining or model choice on thisset.
- Same fixed contour0.5/TTA8/BF16/watermask for equal andhybrid; mapping and
  background asSPEC94. Primary delta mIoU>=.02, event-bootstrapCI95 lower>0,
  no grade perclassdelta<−.02. Bootstrap2000 seed20261003. Persist all counts.
- Meaningful profile isolation/source-year tests and cached-file/model tests
  pass. Sealed standalone GPU scoring queue behind commonflock, after Russian
  work and fullcollectionadmission. CPUcollection usesk8plus, neveromarchy/Mac.
- No claimed globalrank/Prithvi victory from thisseverity comparison; only
  improvement against tested frozenhybrid on held-out retrospectiveMTBS labels.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_fresh_paired_severity.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/prepare_fresh_paired_severity.py --profile retrospective-2016-17 --data-root data-store --source-root /home/mc/fires --prior data-store/prior-footprints.json --out data-store/severity-retrospective-v1`.
- k8plus: `bash scripts/queues/queue_fresh_paired_severity.sh`.
- Receipts include input/model/source hashes, per-event failures/counts,
  limitations: retrospective2016–2017, satellite interpretation, notfieldtruth,
  notSiberia, notprospective, no accuracy evidence from QC/synthetic tests.
