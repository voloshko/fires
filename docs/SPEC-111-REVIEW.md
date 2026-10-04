# SPEC-111 independent review

Read-only code lane accuracy_code_review: APPROVE,12 files reviewed,
zero findings;11 tests, compile, diff and gate passed.
Architecture lane accuracy_arch_review_native: WATCH, no blocker to the
frozen development experiment. Final synthesis COMMENT, not promotion approval.

- Formula/shared-domain check preserve every reference-positive pixel,
  including weak detections. Existing reference FP are also immutable.
- Ideal IoU ceiling is(TP+FN)/(TP+FN+FP), assuming perfect recovery and no
  added FP. Derived ceilings are0.667/0.677/0.673 on the three Russian sets,
 0.867 EMS and0.818 FLOGA; these are not achieved scores.
- Each adapter trains against one base seed; inference uses C1-MM5 plus
  mean-three addition probabilities. The matched post control shares this
  empirical ensemble construction; it does not reproduce averaged single models.
- Bias-4 gives initial q about0.018 and can shift a base probability around
 0.4908 across0.5. Do not claim exact initial identity or prediction neutrality.
- Recall invariance is not improved IoU. The original point/CI/region/miss
  guards and fresh preregistered confirmation remain required for promotion.

No reviewer edited sources, used GPU, changed queues or messaged user threads.
