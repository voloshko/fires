# SPEC-117: Audit CEMS Russian and boreal label source availability

Status: implemented
Requirement: REQ-008
Depends on: SPEC-90, SPEC-96

## Summary

Discovery observed the existing Rapid Mapping API returns only265 records since
March2023. The public mapping portal's historical API instead returned1065 from
2012, no declared Russian records, and four Swedish wildfire activations.
These are already observed discovery counts, not a preregistered accuracy test.
Deliver source-bound metadata evidence and coverage limits before pursuing this
potential source for additional Russian/taiga labels. No model/data promotion.

## Acceptance Criteria

- Fetch public Rapid Mapping and historical portal activation metadata on
  k8plus, freeze raw bytes/request URLs/time/status and SHA256. Authenticate
  declared count, complete pagination, unique codes, list/result schema and
  normalized country/category fields. Missing/partial/error != zero coverage.
- Current endpoint dashboard-api/public-activations-info and historical
  mapping.emergency.copernicus.eu/activations/api/activations. Historical filter
  activationTime2012-01 through2026-10-04. Report actual endpoint bounds rather
  than claiming every possible CEMS/Charter product is listed.
- Declared Russian match uses only country labels Russia/Russian Federation/
  Россия/Российская Федерация; preserve unknown/empty country records explicitly.
  Wildfire classifier category Wildfire or slugfire. Do not infer country or
  event identity from a name, centroid, image-of-day, thermal detection or smoke.
- Report declared Russian records/wildfires, unknown-country count, candidate
  Canadian/Swedish/Finnish/Norwegian wildfire activations and product/AOI counts.
  Separate activation timestamps from physical ignition; post2015 candidate
  counts are metadata only, not available labels, matched Sentinel pixels or
  independent fires. No fit/eval split, old data/selection or threshold change.
- Preserve negative availability evidence. Do not call no declared record nofire,
  or a small boreal list a valid independent Russian test. Any imagery/vector
  collection/training needs a separate specification with frozen source admission.
- CPU-only k8plus/OMP2/CPU200/MemoryMax4G; no images/vector archives/model/GPU.
  Raw metadata+report<=16MiB and3GiB free floor before every write. Immutable
  outputs, raw byte hashes and no credential/header dumps.
- Tests current/historical schema normalization, missing countries, malformed
  category, unique count/completeness and explicit activation-vs-fire distinction;
  source receipt/report, make gate and per-spec commits.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_cems_source_availability.py -q`.
- k8plus: `.runtime/bin/python scripts/audit_cems_source_availability.py --out research/cems-source-availability-v2`.
- `make gate`.
- non_claims: snapshot of two public activation listings, not all worldwide
  mapping products or fireabsence; no labels/imagery or data split, Russian/
  boreal accuracy, field truth, model promotion or original goal completion.

## Execution notes

V1 snapshot completed265/1065 and raw counts replayed exactly. Independent
review found empty individual country labels should remain unknown, rather
than abort on schema ambiguity. Fix preserves mixed known/unknown countries
and five tests pass. V1 raw/result/seal retained; finalv2 uses a fresh output
and source seal. No changed source filters, metric or data admission.

## Resolution

Implemented metadata audit of two public listings. Finalv2 exit0/4.968s/
323.8MiB peak; five tests/gate pass and1330raw records replayed exactly.
Rapid265 since2023-03-24, historical1065 since2012-04-27; both latest2026-09-15.
No declared Russian records/wildfires. Historical one unknown-country record
retained; four country-tagged Swedish wildfire candidates, three from2015+.
No claim of physical fire counts, boreal ecology, available labels or absence
of Russian fires/mapping elsewhere. Raw pages and oldv1 preserved onk8plus,
compact report/seals onMac. Total786017bytes before tiny replayaudit, below16MiB.
Independent codeAPPROVE after unknown-country fix; architectureWATCH, scoped
finalCOMMENT. PASS receipt source availability only; no model/GPU or new data
admission. Original four-direction goal remains active.
