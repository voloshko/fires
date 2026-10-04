# SPEC-118: Five-initializer matched equal-cost temporal additions

Status: implemented
Requirement: REQ-008
Depends on: SPEC-90, SPEC-95, SPEC-114

## Summary

SPEC114 passes competitor/region/Med/recall criteria but pair-post pooled CI
lower-0.002345 remains inconclusive. Prospectively expand both matched
ensembles from3 to5 initializers, aligning their membership with C1-MM5 anchor.
Reuse immutable completed114 seeds1/2/3; train only new4/5 pair/post. Hypothesis:
additional seed diversity reduces unstable additions. No promised gain, CI
retuning, selective seed/epoch choice or reinterpretation of114 FAIL.

## Acceptance Criteria

- Original114 sources/checkpoints/FAIL immutable. Fixed parent resultSHA
  eddf5ba5df6e6d47265503234a7443d2ee4730df9c3c1eef25eefb30105f51bc
  and parent training recipe a6e432e3f0eeb21ac550027e7b5278d67aa1639ed819ed40c4d53539484ab690.
  Authenticate all6 parent native adapters, full summaries/order/exposure/
  frozen backbone, loss/base/source/fit identity; never relabel114 as118.
- New118 trainer/scorer/contract/smoke/queues namespace. Only new seeds4/5,
  pair/post, frozen C1-MM4/5 six-band UNet/depth7/rank8 RecallExpansion.
  Raw initializer SHAs4=504141c858b50615e0a884dfa512f20360ceb2b86455967cb056279d2497ba8b,
  5=a48abe8b5c98883cf2f60a23eb457e39e5fea7584bafb4266d05eb7bcb16719f.
- Same fit374+309/683IDs/762views,Medweight2,30epochs,batch8,
  AdamW1e-3/decay1e-4/cosine/BF16,fit-only locked base statistics, augmentation,
  equal CE[1,1]+unchanged Dice. Same bias-4 prior/monotonic formula.
  New paired orders/exposures/base digests must match within each seed.
- New native4/5 spec118/loss equal-ce-dice-v1/base/seed/variant/new seal;
  reused1/2/3 require114/loss/parentseal identity and actual hashes. Reject
  wrong family/seed/base/variant/smoke/artifact mixture. Full10-run audit before
  score. New controls have the identical membership/capacity/steps as pair.
- Inference C1-MM5 anchor plus UNIFORM mean5 q, TTA8 onq,threshold.5/BF16,
  same valid/water domain, prove reference positives retained. No selective
  weights, seed subset, threshold or source mask choice after outcomes.
- All old development375windows and original exact eligibility unchanged:
  Russia234/90 delta>=.01/CI lower>0 vs post,C1-MM5,Prithvi;
  eachRussianpopulation point>=-.01 to C1-MM/reserve; EMS/FLOGA point>=-.01
  and CI lower>=-.03 vsC1-MM; no extra missed groups. Bootstrap2000seed20261003.
  Authenticate original comparator bytes/caches/counts and reconstruct outcomes.
- Unit tests family/seed/recipe, five-member mean/formula/monotonicity; actual512
  new initializer GPU smoke bothpair/post FP32/BF16/gradient/freezing/native
  roundtrip before training. Seal source/model/loss/parent lineage before launch.
- k8plus only, oneGPUlease/free22GB, fulltrain18G/CPU400%,score12G/CPU400%.
  Small new adapters/packed outputs<=512MiB/freefloor3GiB. No duplicated base
  weights, new imagery, old deletion or local model/data files.
- If ineligible reject with immutableFAIL. If eligible freeze separately specified
  confirmation with actual118 identity, unchanged unused112selection and original
  source/quality criteria. Do not repurpose deferred116's fixed114 binding.
  Tests, independentreview, receipts, gate and per-spec commits; no fullgoal closure.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_five_seed_additions.py tests/test_equal_cost_additions.py -q`.
- k8plus: `bash scripts/queues/smoke_five_seed_additions.sh`.
- k8plus: `.runtime/bin/python scripts/train_five_seed_additions.py --data-root data-store --reference-root /home/mc/fires/research --runs research/five-seed-additions-v1 --seal-only`.
- k8plus: `bash scripts/queues/queue_five_seed_additions.sh` and
  `bash scripts/queues/queue_score_five_seed_additions.sh`.
- `make gate`.
- non_claims: consumed development; aligning initializer membership is not
  exact per-seed/ensemble probability equivalence, accuracy or causal proof.
  Shared publisher/spatialgroups, no field truth/severity/L2A/globalrank,
  product promotion or original four-direction goal completion.

## Resolution

Implemented full ten-member experiment, eligibility true/allfourguards true.
Fournew full30epoch runs completed, sixparent114states reusedunchanged;
all10native/source/fit/exposure/order/base and750packed confusion rows verified.
Russia234/90 pair0.287230/post0.228190/C1-MM0.219247/Prithvi0.116043;
postdelta+0.059040 CI[0.007709,0.098965], C1delta+0.067983
CI[0.004652,0.116854], Prithvidelta+0.171187 CI[0.123718,0.229808].
Regionalpoint/Med/missedgroup guards pass. FLOGA0.745200 vsC1.746460,
allowedpoint-0.001259/lower-0.003997. No uniformimprovement claim.

Train/score terminalexit0; immutablePASS supports consumeddevelopment eligibility
only. Matchingfivemembership does not prove probabilityequivalence/extraevent
samples or causalvariance explanation. Separatelypreregistered119collection
nowauthenticated and live; its freshquality/sourceadmission notyetcomplete.
No product/worldrank/fieldtruth/severity/L2A or originalgoalcompletion claim.
