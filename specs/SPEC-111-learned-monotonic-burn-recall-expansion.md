# SPEC-111: Learned monotonic burn recall expansion

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-95, SPEC-109

## Summary

SPEC109 adds useful temporal information on Russian development, but its
negative correction loses a weak FLOGA detection and its positive bound limits
recovery of low-reference-probability burns. Learn an addition probability q:
p_new=p_C1+(1-p_C1)*q. This preserves every reference-positive pixel on the
same valid/water domain, while false-positive growth remains an empirical risk.
Unlike rejected fixed fusion, additions are trained using fit labels only.

## Acceptance Criteria

- New module/artifact namespace; original109 sources, weights, caches and FAIL
  immutable. Frozen C1-MM1/2/3 encoders/decoder/BN, six-band fit mean/std, same
  rank8 multi-scale projections and16-channel correction head as109.
  Pair features(pre,post,post-pre); capacity-matched post(post,post,zero).
- Final head weight zero, bias-4 before training, q=sigmoid(z). This is a
  declared small positive prior, not exact zero-logit parity. Training uses
  stable log probabilities: log p_bg=log p_base_bg+logsigmoid(-z),
  log p_fg=logaddexp(log p_base_fg,log p_base_bg+logsigmoid(z)).
  Same CE[.25,1]+Dice; no test-derived threshold, masks or labels changed.
- Scoring uses C1-MM5 mean probability plus mean3 addition probabilities,
  TTA8 averaged on q before ensemble, threshold.5. Prove p_new>=p_C1,
  p_C1=1 stays1, p_C1=0 can recover, and identical validity/water domains
  preserve all C1-positive pixels and thus its detected labelled groups.
  Do not claim additions imply improved IoU or that q is an independent fire model.
- Five-point mechanics before full training: formula/log-prob equivalence,
  finite gradient at initial prior, full freezing/BN, pre dependence of pair
  with nonzero head and post-control pre independence, adapter-only native
  save/load and wrong-base rejection. Actual512 fit-window GPU preflight.
- Full experiment identical data/batches/augmentations to109: fit374+309,
  all683 IDs every epoch, original Med repetition2,30epochs,batch8,
  AdamWlr1e-3/weight_decay1e-4/cosine/BF16, seeds1/2/3, pair/post.
  Keep locked normalization and source/software/initializer seals.
- Reuse the exact development eligibility from109: Russia234/90 pooled gain
  >=.01 and CI lower>0 vs post/C1-MM5/Prithvi; each Russian population
  point delta>=-.01 to C1-MM and original reserve; EMS/FLOGA delta>=-.01
  and CI lower>=-.03 to C1-MM; no extra missed positive groups. All234 Russian
  and141 Med windows are consumed development; no promotion/independence claim.
- k8plus only, per-run GPU lease/free VRAM22GB, host MemoryMax18G,CPU400%.
  New small adapter weights/packed predictions only, <=512MiB outputs and
  free floor3GiB; no original deletion or images/weights on Mac. Full scorer
  must reconstruct comparator counts and authenticate caches/model lineage.
- If eligible, freeze new confirmation under a separate spec before new
  data/predictions; otherwise reject, preserve outputs and unchanged criteria.
  Tests, immutable receipts, gate and SPEC111 commits. No copied full backbones.

## Verification

- k8plus unit tests and real512 preflight commands recorded before launch.
- Full trainer/queue/scorer commands recorded before execution.
- `make gate`.
- non_claims: structural reference preservation is not accuracy; additions
  cannot correct existing reference FP. PANGAEA background unreviewed,
  development consumed, no field truth/severity/L2A useful-gain proof,
  global superiority, product promotion or original goal completion.
