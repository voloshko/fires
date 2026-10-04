# SPEC-107: Russian annotation temporal semantics audit

Status: implemented
Requirement: REQ-008
Depends on: SPEC-92, SPEC-105, SPEC-106

## Summary

SPEC-106 shows negative median dNBR and high post NDVI in all four reserve-
missed western windows. That does not establish wrong labels. Audit the
publisher's intended temporal/source meaning before changing pairing or training.
Existing source_manifest claims Jan1–May15 2020, visual satellite labels,
and no individual fire dates. Deliver a source-cited decision about what those
statements justify; preserve all prior labels, measurements and FAIL receipts.

## Acceptance Criteria

- Read the primary PANGAEA.939429 landing page and its cited methods paper;
  retain exact URLs, access UTC, compact text excerpts and source hashes when
  fetched on k8plus. Any PDF download stays on k8plus, maximum 10 MiB. No
  imagery, FIRMS queries, secrets, GPU runs or changes to old supervised files.
- Separate dataset coverage interval, actual fire dates, observation dates,
  spring versus summer/past-year scars, negative/background completeness,
  annotation method and reliability note. Mark unsupported fields UNKNOWN;
  never turn temporal metadata or source absence into per-polygon dates.
- Inspect the already-downloaded ZIP schema on k8plus and list available date
  attributes, with raw source identity intact. Check whether a dated auxiliary
  file is actually present before recommending a new pairing recipe.
- Link observed diagnostic mismatch to at least two plausible explanations;
  distinguish evidence from inference. Do not declare GT wrong from negative
  dNBR or drop any evaluated groups. Report whether currently frozen pairing
  is supported, contradicted, or only an approximation to source semantics.
- Write docs/research/SPEC-107-temporal-audit.md and append receipt. PASS means
  audit performed and uncertainties named, not that labels/accuracy are validated.
  If primary methods are unavailable, preserve available evidence and report
  INSUFFICIENT_EVIDENCE instead of inventing dates. New data/training needs a
  separate spec; make gate and a SPEC-107 commit are required.

## Verification

- Primary source: https://doi.pangaea.de/10.1594/PANGAEA.939429
- Schema audit on k8plus of data-store/russia/burned_area_2020_topology.zip.
- `make gate`.
- non_claims: no field truth, complete negatives, per-fire date recovery,
  causal proof from observed spectra, new model gain or full goal closure.

## Resolution

Publisher landing page and author archive returned HTTP 200 on k8plus;
compact audit records raw HTML SHA256 and UTC. Methods paper was read via
publisher-provided full text on ResearchGate, sections 2.2–2.4 and 4.
No PDF downloaded. Source interval Jan1–May15 2020 confirmed. Existing ZIP
contains 196673 polygons and only the area attribute, no dates/auxiliary file.

The source used multiple in-period observations; fire timing in the paper
was inferred from active detections or first observed scar with uncertainty.
Our seasonal 2019/2020 pair is an approximation to those observations. This
does not prove erroneous labels or invalidate the preserved comparisons.
The source table and author archive differ in descriptions of derived files;
no replacement or reinterpretation of the frozen raw ZIP was performed.
Next: separately specified small multi-date visibility audit on consumed
missed windows, before a new temporal-data/training recipe. No model promotion.
