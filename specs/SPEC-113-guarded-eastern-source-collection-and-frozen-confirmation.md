# SPEC-113: Guarded eastern source collection and frozen confirmation

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-95, SPEC-111, SPEC-112

## Summary

Implement the conditional data/confirmation path for immutable SPEC112 metadata
selection. Reject an ineligible SPEC111 candidate before any catalogue access,
new output directory or image write. If and only if111 is eligible, bind its
actual artifacts and collect the predefined eastern windows, without retuning.
This spec is not executed image/accuracy evidence merely because code exists.

## Acceptance Criteria

- Exact selection SHA256
  0ab2edea7db92eca8e27da5b179ed5b373314d44f025f3019444be162cb74587,
  SPEC112/72windows43groups/115–150E50–65N. Revalidate all four exclusion
  registries, rawZIP/source hashes and metadata identities before collection.
- Candidate must be completed SPEC111 development, eligible true with all four
  guards true. Recompute the exact eligibility from its frozen counts before
  accepting. Verify scoring/training recipe seals, all6 adapter metadata/
  checkpoints/base hashes/fit identities and completed full summaries; actual
  model hashes and source/preprocessing settings sealed BEFORE catalogue access.
  Reject missing/false/stale/provenance failures before creating outputs.
- Own k8plus data-store/eastern-confirmation-v1 only; PCprimary six-band
  B8A/B12, pre2019/post2020 May16–July15 seasonal pair, oneMGRS/grid51230m,
  SCL invalid0/1/3/8/9/10/11, <=.3 bad perdate, commonvalid>=.5, >=9GTpixels.
  All source polygons in actual frame rasterized, including outside the seedbbox.
  No FIRMS, label/threshold changes or false no-fire interpretation of failures.
- Frozen72 windows, no replacement/additional selection after availability or
  predictions. Actual admission>=50windows/20spatialgroups; otherwise
  INSUFFICIENT_EVIDENCE, no inference. Preserve failed/partial files/reasons.
  Storage<=768MiB inclusive metadata/failures, freefloor3GiB before each write,
  serial or2workers with14MiB reservations, no fullscene download or old deletion.
- Freeze model/manifest/source recipe before new predictions. Same pair/post
  adapters and C1-MM5 anchor+mean3 addition probabilities, TTA8/.5/BF16 and
  identical valid/water masks; comparators C1-MM5, Prithvi officialweights/YAML,
  original reserve3. Candidate fresh delta>=.01/CI lower>0 to C1-MM ANDPrithvi
  ANDmatched post, reserve point degradation<=.01, no extra missed groups.
  Bootstrap2000seed20261003. Report all counts/masked_fraction/population/limits.
- Small weights and packed final predictions; reference float probabilities
  only where needed for C1 anchor, scoped cache output<=256MiB and freefloor3GiB.
  All compute/data k8plus, oneGPU lease/free22GB, CPU source collection<=2cores8G,
  scoring<=4cores12G. No images/weights copied toMac; no product changes.
- Tests ineligible/missing/forged guards stop before external I/O or creation;
  source/identity/hash/admission/budget failures and reference-positive
  preservation; make gate and append immutable receipts. A fresh PASS would
  support this source-selected geography only, not all ecological/severity/L2A
  requirements or original goal closure. Deployment needs a separate spec.

## Verification

- Unit tests and guarded collection/scoring commands recorded before execution.
- k8plus: `.runtime/bin/python -m pytest tests/test_eastern_confirmation.py -q`.
- Conditional k8plus collection: `.runtime/bin/python scripts/collect_eastern_confirmation.py --data-root data-store --selection data-store/eastern-holdout-plan-v1/selection.json --candidate research/recall-expansion-v1/development.json --runs research/recall-expansion-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out data-store/eastern-confirmation-v1`.
- Fresh collection never begins merely on metadata readiness or partial model
  completion; keep the new population untouched when111 is ineligible.
- `make gate`.
- non_claims: shared PANGAEA annotation/spring2020 and spatial groups, not
  physical ignition events/field truth; raw background not reviewed absence;
  no globalrank, product promotion, grade/L2A acceptance or full goal completion.
