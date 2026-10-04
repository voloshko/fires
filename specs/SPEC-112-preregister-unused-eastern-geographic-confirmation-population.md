# SPEC-112: Preregister unused eastern geographic confirmation population

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-95, SPEC-105, SPEC-111

## Summary

All previous Russian evaluation/calibration/Western populations are consumed.
Freeze a metadata-only unused geographic population before any new images or
predictions, while SPEC111 trains. This spec delivers selection and lineage,
not collected data or independent model acceptance. Subsequent collection and
scoring require an eligible frozen candidate and a separate executable spec.

## Acceptance Criteria

- Same frozen PANGAEA ZIP SHA from92/105, raw valid WKB IDs; fixed bbox
 115–150E/50–65N. Read all valid polygons intersecting the box; retain tiny
 polygons for grouping. Group within this preregistered geographic population
 with EPSG3576 connected20km graph, exactly the105 convention, not physical
 fire IDs. Area≥1ha for frame seeds only; no model/imagery selection.
- Four exclusion registries: all2658 SPEC96 footprints, all447 Russian frames,
  all173 covered severity frames, ALL96 consumed Western frames. Exact existing
  SHA/spec/count/readiness/unique ID/bbox/input-hash checks before spatial work.
  Any contour or proposed512/30m frame within20km of an old footprint purges
  its ENTIRE candidate group. No partial group salvage after outcomes.
- Frozen deterministic SHA('spec112:'+sample_id/group_id) order and area bins
  1–25/25–250/>250ha, round robin groups before additional frames, max4/group,
  cap72. Ready metadata population needs≥50 windows and≥20 groups, otherwise
  INSUFFICIENT_EVIDENCE; do not lower thresholds or change bbox to pass.
- Same frame/grid/MGRS source constraints as105, but no scenes downloaded,
  prediction caches read or labels inspected beyond polygon metadata. Plan
  own k8plus data-store/eastern-holdout-plan-v1/selection.json, compact≤512KiB,
  CPUQuota200%/MemoryMax8G. No old deletion or mutation; no arrays/weights onMac.
- Record source/registry/code hashes, removed groups/counts and full selected
  frame identities. Re-execution cannot change the immutable plan. Source
  polygons/area metadata select geography; catalogue availability not examined.
- Tests missing any exclusion source/record/stale hash, group purge, repeated
  group allocation and deterministic area order. make gate/receipt/commit.
  If metadata-ready, record constraints for the later collection spec: PC
  primary, commonvalid/SCL/source interval as105,≤768MiB imagery/free3GiB,
  admission≥50/20 and model hashes fixed before images/predictions. These
  constraints do not authorize collection or validation within this spec.
- non_claims: shared publisher/season, independent unused geography only;
  no field truth, exact ignition dates, new data admission, model superiority,
  ecological transfer everywhere, grade/L2A acceptance or full goal completion.

## Verification

- k8plus CPU planner and unit-test commands recorded before execution.
- `make gate`.
