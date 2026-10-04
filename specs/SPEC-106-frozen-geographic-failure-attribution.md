# SPEC-106: Frozen geographic failure attribution

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-95, SPEC-105

## Summary

SPEC-105 completed 96 windows / 37 groups. Reserve gained 0.036730 IoU over
C1-MM but its CI crossed zero, with two missed positive groups versus none.
Before choosing a new training recipe, explain the error concentration and
observable spectral/date conditions from completed outputs. This is a CPU
diagnostic on consumed evaluation data, never a new confirmation or tuning run.

## Acceptance Criteria

- Bind the exact SPEC-105 parent result SHA256
  749699f3f6e01f0a0a6576afb496c891c4bc33c05ad1cca4087585ed5628cb2c,
  manifest, frozen recipe/source hashes, input rasters and all three signed
  prediction caches. Missing/changed caches fail before any forward pass;
  no training, GPU inference, imagery downloads or changes to old artifacts.
- Reconstruct every parent per-window TP/FP/FN on identical valid/water masks.
  Any disagreement aborts. Report all 37 group totals, per-group IoU and the
  whole-population candidate-minus-C1-MM delta after dropping each one group.
  This leave-one-group-out table diagnoses influence; it is not a second CI,
  unique burned area or a rule for removing inconvenient groups.
- For every window, summarize valid labelled burn, reserve-only false positives
  and C1-MM-only false positives. Keep .5 threshold and original water rule.
  Record count and q10/median/q90 of pre/post NBR, dNBR, pre/post NDVI and
  post NDWI, with explicit undefined ratios excluded and an empty set reported
  as null rather than zero. Six-band order is B2/B3/B4/B8A/B11/B12.
- For all missed reserve groups/windows, report parent truth/TP/FN, probability
  maxima on observed truth, truth removed by the water rule, acquisition dates
  and days after the upper source interval boundary. No invented ignition day,
  reviewed-negative status, cloud unmasking or FIRMS queries.
- Tests: wrong/misaligned identity or counts abort; dropping a group uses sums
  of all its windows; zero spectral denominator/empty subsets stay explicit;
  cloud/invalid/water pixels do not become false-positive diagnostic targets.
  Run make gate, append immutable result/PASS receipt for diagnostic execution
  only, then resolve SPEC-106 and commit. New model/data hypotheses need new specs.
- All raster/array computation on k8plus, CPUQuota200%, MemoryMax6G. Only
  compact JSON (<2 MiB) copied to Mac. Source/results stay immutable. If inputs
  do not reconstruct, publish INSUFFICIENT_EVIDENCE and fix the proven binding
  problem before further training; do not use unbound diagnostics for selection.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_geographic_failure_report.py -q'`
- k8plus: `.runtime/bin/python scripts/report_geographic_failures.py --data-root data-store --manifest data-store/western-holdout-v1/manifest.json --parent research/western-holdout-v1-result.json --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out research/western-holdout-v1-diagnostics.json`
- `make gate`.
- non_claims: post-result diagnosis, not causal proof, field truth, new model
  improvement, independent selection evidence, global superiority or goal closure.
