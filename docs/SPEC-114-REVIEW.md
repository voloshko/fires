# SPEC-114 equal-cost training review

Independent code lane accuracy_code_review: APPROVE, zero issues.
Independent architecture lane accuracy_arch_review_native: WATCH, no blocker
for the declared training ablation after real512 preflight and recipe seal.
Final synthesis COMMENT. Mechanics/review do not establish accuracy acceptance.

- Equal-cost CE increases relative background pressure. Weighted-mean CE's
  denominator changes as well: with foreground fraction f, background CE
  gradients scale by 1+3f, foreground gradients by .25+.75f against111.
  There is no uniformly fourfold background-gradient claim. Dice is unchanged,
  so outcomes also reflect the changed balance between CE and Dice.
- Pair/post control retains pool, batches, augmentation, initializers, frozen
  states and architecture. Attribution against111 requires cross-experiment
  trace/source checks and paired metrics; eligibility alone cannot prove cause.
- Monotonicity preserves existing reference FP. Training can reduce new FP,
  but all original IoU/CI, geography and missed-group criteria still apply.
- SPEC114/loss identity and its artifact/cache namespaces are separate.
  All current datasets are consumed development. SPEC112 remains untouched;
  any eligible successor needs a separate confirmation specification.

Nine unit tests and real512 FP32/BF16 pair/post mechanical checks passed.
First systemd launcher failed127 before any script/GPU work because its working
directory was unset; preserved separately. Corrected smoke-v2 finished exit0,
5.316seconds/1.6GiB host peak, without overwriting any successful output.
Both reviewers were read-only with no GPU/network or queue changes.
