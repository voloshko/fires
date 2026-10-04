# SPEC-110: MTBS legend and severity error attribution

Status: active
Requirement: REQ-008
Depends on: SPEC-94, SPEC-100, SPEC-107

## Summary

Official MTBS FAQ identifies raw0 as outside assessed mapping/background,
raw5 as burned increased greenness, and raw1 as unburned-to-low mixture.
Existing severity evaluation maps0/1/5 to class0. Audit what that benchmark
convention contributes to errors before a new severity model. Do not replace
its primary metric with an easier assessment-only score or relabel old FAILs.

## Acceptance Criteria

- Source-linked report cites https://www.mtbs.gov/faqs and
  https://www.mtbs.gov/product-descriptions. Separate benchmark class0 from
  physical evidence of no fire; raw0 is not reviewed absence, raw5 is not
  unburned, raw1 is mixed. Original archived label conventions remain intact.
- On k8plus, bind immutable SPEC94 development172 and SPEC100 covered173
  parents, manifests, all input hashes and already signed probability/boundary
  caches. No new training, inference, imagery download, old-file changes or
  GPU use. Missing/unbound cache fails before any network forward.
- Verify .sev.tif values originate from unremapped MTBS0–6 ImageServer export.
  Report observed/invalid histograms per raw value, unknown codes separately.
  Do not turn absent/unprocessed areas into known negatives.
- Reconstruct EVERY archived 4x4 confusion for equal/balanced/hybrid where
  present. Then partition legacy damage predictions by original raw0,1,2,3,4,5,6
  (and other codes), with all totals recombining exactly. Record per-window,
  totals, observed fraction, source-assessed fraction, and increased-greenness
  fraction. No threshold, water rule, contour or severity class is changed.
- Explain whether legacy false-grade penalties arise from outside assessment,
  mixed unburned-low, increased greenness, or errors on graded2/3/4. This is
  descriptive consumed-data attribution, not causal proof or new accuracy.
  Do NOT compute a new PASS by masking away errors, change old acceptance,
  claim all raw0 pixels are annotation errors, or train on a raw5-negative label.
- Tests per-code decomposition/recombination, ignored validity, missing/unknown
  values and impossible geometry/predictions. CPUQuota200%,MemoryMax6G;
  compact result<=2MiB copied locally, no images/weights. Append receipt,
  make gate, source-linked report and SPEC110 commit; no promotion from audit.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_mtbs_legend_audit.py -q`.
- CPU audit command recorded before execution, writes
  research/mtbs-legend-audit-v1.json from frozen existing artifacts.
- k8plus: `.runtime/bin/python scripts/audit_mtbs_legend.py --data-root data-store --reference-root /home/mc/fires/research --out research/mtbs-legend-audit-v1.json`.
- `make gate`.
- non_claims: benchmark convention audit, not corrected independent model
  superiority, true burned area/field severity, reclassified labels, physical
  fire absence outside mapped areas or original goal completion.
