# SPEC-114: Equal-cost background supervision for monotonic temporal additions

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-95, SPEC-111

## Summary

SPEC111 improves Russian point IoU but adds14028 false positives for370 new
true pixels on FLOGA. Its inherited CE weights background .25/fire1.
Test equal-cost CE[1,1] plus the identical Dice term on stable expanded
log probabilities. Hypothesis: reduce background additions without losing
learned temporal recovery. This is a new training ablation, not post-result
inference calibration. The suspected loss mechanism is not proven causality.

## Acceptance Criteria

- Preserve original111 artifacts/code/FAIL. New114 trainer/scorer/smoke/queues
  and artifact namespace. Reuse immutable RecallExpansion rank8 architecture,
  frozen C1 encoders/decoder/BN, bias-4 prior and p_new=p_C1+(1-p_C1)*q.
- Sole numerical intervention: cross_entropy with equal class weights, same
  ignore_index=-1 and Dice formula/smoothing1. Compute FP32 from stable log
  probabilities. Reject no-supervision batches; ignored pixels have zero loss
  gradient. Demonstrate stronger background pressure relative to positive
  pressure on a controlled mixed-label example, without claiming actual gain.
- Same683 fit IDs,762 views/epoch, Med repetition2, seeds1/2/3 pair/post,
 30epochs,batch8,AdamW1e-3/decay1e-4/cosine/BF16, initializer hashes, fit-only
  locked statistics, batch orders/augmentation. No labels/source/masks changed.
- Native adapters explicitly SPEC114 with loss identity equal-ce-dice-v1;
  reject111 artifacts or mismatched loss/base/variant. Seal all source/model/
  initializer/normalization/loss recipe before full training.
- Actual512 GPU smoke both variants, FP32/BF16 formula parity, finite nonzero
  gradient, frozen weights/BN, native roundtrip. Unit tests for loss formula,
  unknown labels/gradient and wrong artifact identity. No accuracy smoke claim.
- Unchanged full development eligibility: Russia234/90 delta>=.01 and CI
  lower>0 vs matchedpost/C1-MM5/Prithvi; each Russian population point>=-.01
  to C1-MM and original reserve; EMS/FLOGA point>=-.01 and CI lower>=-.03
  to C1-MM; no extra missed groups. Same TTA8/.5/BF16, common masks,
  group bootstrap2000seed20261003. Report111 numbers only as consumed context.
- Reconstruct all counts/model lineage; matched complete six-run execution.
  If ineligible reject; if eligible preregister a separate successor confirmation
  spec. SPEC112 remains unused; do not bypass deferred113's111-only binding.
- All compute/data k8plus. Shared GPU lock per run, free VRAM22GB, train18G/
  CPU400%,score12G/CPU400%, output<=512MiB/freefloor3GiB. No new imagery,
  copied backbones, deletion, inference threshold tuning or product changes.
- Independent review, tests, immutable receipts, make gate and per-spec commits.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_equal_cost_additions.py tests/test_recall_expansion.py -q`.
- k8plus: `bash scripts/queues/smoke_equal_cost_additions.sh`.
- k8plus: `.runtime/bin/python scripts/train_equal_cost_additions.py --data-root data-store --reference-root /home/mc/fires/research --runs research/equal-cost-additions-v1 --seal-only`.
- k8plus: `bash scripts/queues/queue_equal_cost_additions.sh` and
  `bash scripts/queues/queue_score_equal_cost_additions.sh`.
- `make gate`.
- non_claims: consumed development, published background not reviewed absence;
  no new independent geography/field truth/severity/L2A/global ranking,
  product promotion or completion of the original four-direction objective.

## Resolution

Rejected full eligibility without changing thresholds. All6 full30epoch runs,
within/cross111 data/order/exposure/base identity and750 packed confusion
counts verified. Russia234/90 pairIoU0.292350/C1-MM0.219247/Prithvi0.116043;
pair-C1 delta+0.073103 CI95[0.000426,0.129729], pair-Prithvi+0.176307
CI95[0.131200,0.231691]. Pair-post+0.070378 CI95[-0.002345,0.126803]
fails positive lower bound. Regional points, Med and missed-group guards pass.
FLOGA0.742196 vs C1-MM0.746460, allowed delta-0.004264 lower-0.014916.

Controlled loss ablation FLOGA+0.017909 CI[0.000817,0.056110] vs111, while
Russian calibration-0.038685 CI[-0.064883,-0.007676]. Added FLOGAFP reduced
14028->2592; not universally better loss. Train/score exit0 and no retries.
Immutable FAIL receipt SPEC-114-REPLAY-001. Competitor gains are consumed
regional development; no independent confirmation/promotion or original goal
closure. Conditional116 skipped without collecting images;112 stays unused.
