# SPEC-115: Audit unresolved HLS publisher metadata and archival label reuse

Status: active
Requirement: REQ-008
Depends on: SPEC-96, SPEC-97

## Summary

SPEC96 has17 unresolved HLS fit scenes. Publisher indices name image/mask files
but contain no event IDs. Test whether unchanged source labels repeat across
acquisition dates or correspond geometrically to older official MTBS contours
outside the original450day rule. This is a source audit, not an inferred new
fire date or a replacement of the original source-admission criteria.

## Acceptance Criteria

- Authenticate original540 source/label byte hashes and official MTBS vector
  components against immutable SPEC96 registry SHA
  86f2210603e5921d384c62a52e90167ae64abbad3473bc1cce686ea7bbad3ed2.
  Exactly17 unresolved records. Preserve registry/proof/L2A masks/recipes.
- Bind publisher README and training/validation index bytes, record index
  columns and raster tags/band descriptions for each unresolved scene.
  Missing publisher event IDs remain unavailable, never manufactured.
- Exact label reuse requires same CRS, affine, shape and binary labels AND
  identical source-valid pixel domain; report matching source acquisition dates
  and existing registry association, without claiming physical burn time.
  Compare all540 fit scenes; include no evaluation labels or model predictions.
- Report all prior-ignition official perimeter candidates intersecting each
  frame, including age>450 days. Rasterize per-event union on original grid,
  record GT-positive overlap numerator/denominator and official ignition.
  Unknown ignition excluded and counted explicitly. No relabeling or date
  substitution. Old .95/.05 association checks are descriptive only here.
- Distinguish exact reused target, older geometric candidate, ambiguity and
  absent candidate. Retain all17 outputs, including negative observations.
  Audit cannot establish complete original event retention, label fire date,
  retained protected pixels or actual L2A accuracy. No auto-ready transition.
- CPU k8plus only, OMP2/MemoryMax6G/CPU200%, compact JSON<=8MiB and3GiB free
  floor before write. No images download, model/GPU work or old deletion.
- Test alignment/valid-domain checks, future/unknown-ignition exclusion and
  reused-mask non-attribution. Source-bound receipt, report, gate, per-spec commit.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_hls_label_provenance.py -q`.
- k8plus: `.runtime/bin/python scripts/audit_hls_label_provenance.py --source-root /home/mc/fires --registry data-store/hls-events.json --out research/hls-label-provenance-v1.json`.
- `make gate`.
- non_claims: no publisher lookup/physical ignition inference, new label IDs,
  change to SPEC96/97/93 admission, new data/predictions, competitor improvement
  or original four-direction goal completion.
