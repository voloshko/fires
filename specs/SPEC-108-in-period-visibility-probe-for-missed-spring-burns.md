# SPEC-108: In-period visibility probe for missed spring burns

Status: active
Requirement: REQ-008
Depends on: SPEC-93, SPEC-105, SPEC-106, SPEC-107

## Summary

SPEC-106 found four missed windows with green post scenes and negative dNBR.
SPEC-107 confirms labels refer to Jan1–May15 2020, with no polygon fire dates.
Test whether in-period observations show additional spectral separation from
published background. This is a small visibility experiment on consumed data,
not validation of a new model or justification to change old labels/results.

## Acceptance Criteria

- Bind SPEC-105 result SHA256
  749699f3f6e01f0a0a6576afb496c891c4bc33c05ad1cca4087585ed5628cb2c,
  its manifest and input hashes. Exactly its four reserve-missed windows,
  ordered by sample_id; original grid512/30m, labels, common-valid mask and MGRS.
- Freeze metadata-only plan before raster reads. PC sentinel-2-l2a primary;
  four inclusive date bins: March1–15, March16–31, April1–30, May1–15 2020.
  In each bin retain at most three same-MGRS candidates, sorted by tile cloud
  fraction then acquisition time/id. Skip unknown/nonfinite tile cloud or >80%.
  Plan records item IDs/dates/MGRS/cloud/baseline, no signed asset URLs/tokens.
- Read SCL candidates in frozen order, choose first with strict valid fraction
  >=.5; if none, mark no observation. Fetch six bands only for that one choice,
  at most16 full windows total. Explicit baseline/scale/offset/B8A validation;
  strict invalid SCL0/1/3/8/9/10/11. No cloud unmasking or FIRMS queries.
- Usable observation needs full current valid>=.5, common old/current valid
  >=.5, >=9 common labelled pixels and >=9 published-background pixels.
  Keep other outcomes and reasons, never reinterpret missing observations as
  no fire. No fallback based on burn spectra or model predictions.
- For each usable view, report old/current median NBR and NDVI on identical
  common GT and published-background pixels, q10/median/q90 and masked_fraction.
  Contrast is GT median NBR minus background median NBR, descriptive only.
  A view supplies extra separation if current contrast<=-.1 and old-minus-current
  contrast>=.1. Freeze these .1 values before outcomes, no threshold sweep.
- A frame is temporally assessable with >=2 usable dates separated by >=14 days.
  At least two assessable frames required or INSUFFICIENT_EVIDENCE. Support
  multi-date follow-up only if >=2 assessable frames have >=1 extra-separation
  view each. Otherwise reject this visibility hypothesis; do not close the
  original accuracy goal on a diagnostic PASS or these selected four windows.
- New files only own k8plus data-store/spring-visibility-v1, total<=128MiB
  inclusive partial files/metadata/failures, free floor3GiB before each write,
  serial collection with8MiB per-view reservation. No full scenes, original
  deletion, GPU inference/training, imagery copied to Mac or label changes.
- Tests source/date/tile ordering and missing metadata, same-pixel comparison
  (mask changes cannot fake separation), absent/empty observations, budget
  and frozen acceptance. Append immutable plan/manifest/result and receipt,
  make gate, resolve and commit per spec. New training requires another spec.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_spring_visibility.py -q'`
- k8plus: `.runtime/bin/python scripts/probe_spring_visibility.py --data-root data-store --manifest data-store/western-holdout-v1/manifest.json --parent research/western-holdout-v1-result.json --out data-store/spring-visibility-v1`.
- `make gate`.
- non_claims: post-result selected examples, not new independent accuracy;
  no exact ignition day, field truth, reviewed negative background, causal
  proof of regrowth/misannotation, full interval coverage or goal completion.

## Execution repair

The initial unit terminated with exit1 after writing eight image/SCL pairs:
JSON could not serialize a NumPy boolean. The partial result and original
source snapshots are preserved with an execution FAIL receipt. Correct the
boolean boundary and serialize before creating a destination file. Recovery
uses the exact original plan SHA256
4161f1ab66986c09378ed631292e018cb2c1cffbc6eac65af283bca479bf6320,
unchanged dependencies and source-verified retained pixels; it writes
result-recovered-v2.json, never replaces plan/result/pixels. No date, selection,
observation threshold or scientific acceptance is changed. Recovery is the
same command with --recover; a separate unit avoids restarting the old handle.
