# SPEC-116: Frozen eastern confirmation for equal-cost temporal candidate

Status: active
Requirement: REQ-008
Depends on: SPEC-112, SPEC-114, SPEC-113

## Summary

Prepare the previously unused SPEC112 selection for a genuinely eligible frozen
SPEC114 model. SPEC113 remains deferred and111-only; its artifacts and source
remain unchanged. This successor exists before114 outcomes, with identical
source admission and fresh accuracy criteria, and no data until eligibility.

## Acceptance Criteria

- Frozen SPEC112 selection SHA
  0ab2edea7db92eca8e27da5b179ed5b373314d44f025f3019444be162cb74587,
 72windows43groups,115–150E50–65N; authenticate all four exclusion registries,
  ZIP annotation bytes and selected source metadata before collection.
- Candidate completed SPEC114 development, eligible is true and exact four
  guards true. Recompute all development decisions/counts. Fixed training
  seal a6e432e3f0eeb21ac550027e7b5278d67aa1639ed819ed40c4d53539484ab690
  and loss equal-ce-dice-v1/CE[1,1]; scoring/training sources unchanged.
  Authenticate six native adapters, source/initializer/fit metadata, completed
  summaries/order/exposure/base freezing, cached candidate/comparator counts,
  C1-MM5/reserve3/Prithvi actual weights/configs BEFORE external I/O or mkdir.
  Reject111 artifacts, partial114, forged flags/counts, stale sources or caches.
- New116 collector/scorer/validator namespace; reuse immutable storage/source
  helpers from113 explicitly hash-bound. Data only data-store/eastern-confirmation-v2,
  results only research/eastern-confirmation-v2/result.json, without overwriting
 112selection/111/113/114 outputs. Fixed72windows, no replacement or reselection.
- PCprimary B8A/B12 six bands, pre2019/post2020 May16–July15 annual-interval
  pair, sameMGRS/grid51230m. SCL invalid0/1/3/8/9/10/11, <=.3 bad/date,
  commonvalid>=.5, retainedGT>=9pixels. Rasterize all polygons in actual frame.
  No FIRMS, inferred fire dates, label/threshold/mask changes.
- Source admission>=50windows/20spatialgroups, otherwise insufficient/no inference.
  Preserve every failure/partial file. Storage<=768MiB including metadata,
  14MiB per-worker reservation/two workers,3GiB free floor before writes.
- Freeze candidate/source/model/loss manifest binding before fresh predictions.
  Same mean3 additions around C1-MM5 anchor, TTA8/.5/BF16, water/valid domain.
  Fresh delta>=.01 andCI lower>0 versus matchedpost,C1-MM5,Prithvi, reserve
  point>=-.01; no extra missed groups vsC1-MM. Bootstrap2000seed20261003.
  Report confusion counts, masked fraction, population and all limits.
- Reference probability caches/packed final masks<=256MiB,freefloor3GiB;
  collection CPU2/8G, scoring CPU4/12G; all k8plus, sharedGPUlease/free22GB.
  CPUnegative proof while114trains, no competing GPU/model inference.
- Test each invalid prerequisite stops before catalogue/output; forged decision
  rejection, wrong spec/loss/artifact identity, source admission/storage guards.
  Independent review, immutable receipts, gate and per-spec commits. Eligible
  success path remains unexecuted until actually admitted/scored.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_equal_cost_eastern_confirmation.py tests/test_eastern_confirmation.py -q`.
- Conditional collector: `.runtime/bin/python scripts/collect_equal_cost_eastern.py --data-root data-store --selection data-store/eastern-holdout-plan-v1/selection.json --candidate research/equal-cost-additions-v1/development.json --runs research/equal-cost-additions-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out data-store/eastern-confirmation-v2`.
- Conditional scorer: `.runtime/bin/python scripts/score_equal_cost_eastern.py --data-root data-store --selection data-store/eastern-holdout-plan-v1/selection.json --candidate research/equal-cost-additions-v1/development.json --runs research/equal-cost-additions-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --manifest data-store/eastern-confirmation-v2/manifest.json --out research/eastern-confirmation-v2/result.json`.
- `make gate`.
- non_claims: shared PANGAEA publisher/spring2020, groups not physicalfireIDs;
  background not reviewed fireabsence; no field truth, worldranking, severity/
  L2A success, product promotion or full original goal completion.
