# SPEC-98: Fresh confirmation of equal-weight paired severity

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-94, SPEC-95

## Summary

SPEC94 equal paired head achieved mIoU0.3672 versus hybrid0.3089 on an adaptive
2022+ development set. Weighted hypothesis was rejected. Freeze the existing
three equal checkpoints and evaluate previously unused 2024 MTBS events. No
training or choosing another model on this set. This tests fresh satellite-label
agreement; it cannot establish field accuracy or a global leaderboard rank.

## Acceptance Criteria

- Freeze a plan before downloading predictions: official MTBS2024 CONUS events,
  dimensions<=14km, western forest/mountain states CA,OR,WA,ID,MT,CO,WY,UT,NV,AZ,NM.
  Order SHA256('spec98:'+event_id), at most180 candidate events. Exclude every
  old MTBS fresh1–4 event and prior2658frame intersection, using fixed512x512
  30m centroid frames in the event UTM CRS. Also exclude intersections among
  selected new frames. All exclusions/selection/QC failures are retained.
- Select postHLS30–150days after ignition by lowest actual window Fmask bad
  fraction, then acquisition/id, cloud/cirrus/adjacent/shadow/snow/nodata<10%.
  Pre strictly before ignition within90days, sameMGRS/grid, common-valid pixels.
  Download MTBS2024 severity by nearest-neighbor on that exact grid and verify
  response grid/CRS/shape, classes0–6, source URL and input hashes.
- At least100 previously unused evaluation event groups, with all3 severity
  classes represented, or INSUFFICIENT_EVIDENCE (no model conclusion).
  Inputs/output live only under the own data-store root on k8plus; minimum8GiB
  free before each new event; workers2; no old files overwritten/deleted.
- Existing SPEC94 equal-s1/2/3 must exactly match model hashes, manifest hash,
  recipe and summary archived in SPEC94 result. C1-MM boundary hashes also match.
  The candidate cannot retrain. No dependence on new labels for normalization.
- TTA8/BF16, same fixedC1-MM boundary0.5 and samewatermask for equal and hybrid.
  Labels0/1/5 known background,6ignored,2/3/4=>1/2/3 as SPEC94.
  Primary acceptance: equal−hybrid mIoU>=.02, event-bootstrapCI95 lower>0,
  no perclass delta<−.02. Bootstrap2000 seed20261003, all confusion/counts saved.
  This frozen rule is independent of the already observed SPEC94 result.
- Small QC run may validate data plumbing but never model acceptance.
  Manifest/result/receipts append-only; cache bound to new inputs and source.
  Meaningful tests reject old/spatially overlapping events and stale model hashes.

## Verification

- `python -m pytest tests/test_fresh_paired_severity.py -q`; `make gate`.
- k8plus: `.runtime/bin/python scripts/prepare_fresh_paired_severity.py --data-root data-store --source-root /home/mc/fires --prior data-store/prior-footprints.json --out data-store/severity-fresh-v1`.
- k8plus: `.runtime/bin/python scripts/score_fresh_paired_severity.py --data-root data-store --manifest data-store/severity-fresh-v1/manifest.json --runs research/paired-severity-v1 --development-result research/paired-severity-v1-result.json --reference-root /home/mc/fires/research --out research/fresh-paired-severity-v1-result.json`.
- Serial GPU scoring behind research/accuracy-gpu.lock after Russian queue;
  CPU collection systemd quota200%, MemoryMax8G. No omarchy computation.
- non_claims: published MTBS agreement only, CONUS2024 only, no field truth,
  no Siberian severity validation, no superiority over models not compared.
