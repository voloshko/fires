# SPEC-109: Post-anchored learned temporal correction

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-95, SPEC-105, SPEC-108

## Summary

The unconstrained paired network loses high-confidence C1-MM detections on
small spring burns. SPEC-108 did not establish that date replacement resolves
those losses. Learn temporal corrections anchored to frozen C1-MM, instead of
retraining its post classifier. Small adapter checkpoints also avoid copying
~498MB backbones on k8plus, which has about4GiB free. No ImageNet initializer,
fixed threshold fusion or inference-only dNBR rule is introduced.

## Acceptance Criteria

- Frozen C1-MM seeds1/2/3 from SPEC-92 parent hashes; preserve original six-band
  mean/std and all backbone parameters/BN buffers. Shared encoder evaluates
  dates separately under eval/no-grad. Zero final correction head reproduces
  post-only logits exactly at initialization, for arbitrary pre input.
- At each of seven encoder scales project concat(pre,post,post-pre) to8 channels
  with1x1 Conv+ReLU; resize to original frame, concatenate,3x3 Conv to16+ReLU,
  final1x1 Conv to1 initialized zero. Control substitutes(post,post,zero), same
  trainable capacity and data. No batch normalization in the adapter.
- Scalar correction =4*tanh(z) for z>=0,2*tanh(z) for z<0, range[-2,+4].
  Add it only to the burn logit. Raw post logit margin>=2 cannot become negative;
  this is a confidence invariant, not a guarantee of retaining every weak fire.
  Global scoring anchors original C1-MM5 mean probability with mean3 corrections
  by odds update p*exp(delta)/(1-p+p*exp(delta)); preserve endpoint probabilities.
- Before full training, CPU tests and actual real512 GPU preflight must prove
  zero-head parity (FP32/BF16), correction gradients finite/nonzero, frozen
  backbone/BN, pre-date dependence after nonzero head, immutable adapter-only
  save/load binding, confidence bounds and both empty/invalid input rejection.
  Smoke and calibration mechanics never establish accuracy.
- Full paired/control experiment: existing paired fit374 plus Russian fit309,
  all683 unique fit IDs every epoch, original Mediterranean weights via repetition;
  no evaluation/calibration/western IDs fitted or mined. Same ordered batches
  and augmentations, locked C1-MM mean/std,30epochs,batch8,AdamWlr1e-3,
  weight_decay1e-4,cosine, BF16, original CE[.25,1]+Dice. Seeds1/2/3.
  Train adapter only; no full backbone export or normalization recomputation.
- Preseal all inputs, source/software/config/initializer hashes. Only k8plus,
  one GPU job via research/accuracy-gpu.lock, free VRAM>=22000MiB. Full runs
  MemoryMax18G/CPUQuota400%. Check free disk>=3GiB before artifact writes;
  adapter-only checkpoints, total new outputs<=512MiB, no old asset deletion.
- Development: unchanged old Russian evaluation77, consumed calibration61,
  consumed Western96, EMS60/FLOGA81. Same valid/water masks,TTA8,threshold.5,
  bootstrap2000seed20261003, all model/cache identities and per-window counts.
  Pool Russian populations for eligibility only; report each separately.
  Eligible iff pooled gain>=.01 and CI lower>0 vs C1-MM5 AND Prithvi,
  gain>=.01/CI lower>0 vs capacity-matched post control, each Russian population
  delta>=-.01 to C1-MM and original reserve, each Mediterranean delta>=-.01
  and CI lower>=-.03 to C1-MM, no extra missed positive groups in any dataset.
- No promotion from development. If eligible, preregister new geographic/date
  confirmation under a separate spec before imagery/predictions, with original
  competitor CI/minimum-population safeguards. If ineligible reject and keep
  receipts/weights. Do not tune architecture, correction bounds or thresholds.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_temporal_adapter.py -q`.
- Real GPU preflight will write research/temporal-adapter-smoke-v1.json and
  small native checkpoint, binding exact base hash and fit sample hashes.
- k8plus: `bash scripts/queues/smoke_temporal_adapter.sh` (shared lease and
  22GB free-memory admission before the source-bound real512 smoke).
- Full trainer/scorer/queue commands recorded before launch; make gate required.
- k8plus: `.runtime/bin/python scripts/train_temporal_adapter.py --data-root data-store --reference-root /home/mc/fires/research --runs research/temporal-adapter-v1 --seal-only`.
- k8plus: `bash scripts/queues/queue_temporal_adapter.sh` (three pair/post
  seed pairs, original fit exposures, per-run shared GPU lease).
- k8plus: `bash scripts/queues/queue_score_temporal_adapter.sh` (wait for six
  completed runs, then the shared GPU lease; save lossless packed binary
  predictions and source-bound counts under the512MiB output budget).
- non_claims: structural/smoke guarantees are not accuracy; all existing
  development populations consumed, PANGAEA background unreviewed, no physical
  fire IDs/field truth/severity/L2A useful-gain proof or original goal closure.

## Resolution

All six runs completed30epochs/683 unique fit IDs; every original view exposed
30 times, Mediterranean60, total22,860 exposures per run. Pair/post epoch
orders and frozen base digests matched in all three seeds. Training/scoring
exit0, eleven tests passed. Checkpoints428,778 bytes each; all outputs28,515,015
bytes, within512MiB, free disk4,273,479,680 bytes. Source-bound packed-mask
reconstruction exactly reproduced counts, decisions and bootstrap intervals.

Pooled Russia234windows/90groups: pair0.249756, post0.233591,
C1-MM0.219247, Prithvi0.116043. Deltas +0.016165 to post
CI[0.009942,0.026590], +0.030510 to C1-MM CI[0.002114,0.046720],
+0.133713 to Prithvi CI[0.063036,0.215316]. Temporal information is useful
on this consumed development pool, but the full recipe is rejected.

Russian calibration pair0.262026 versus original reserve0.304575
(delta-0.042548); Western pair0.198072 versus reserve0.220628
(delta-0.022556). Both exceed permitted0.01 degradation. FLOGA pair0.727710
versus C1-MM0.746460 (delta-0.018750, CI[-0.083773,0.018611]); one
additional missed positive group versus C1-MM0. Three guards fail. Retain
FAIL, all outputs and original thresholds; no deployment or fresh-confirmation
claim. Strong-reference detections were preserved but weak detections still
can be lost, as independent architecture review predicted.
