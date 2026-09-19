# Independent review: hypothesis campaign

Reviewed worktree `codex/fire-model-hypotheses`, implementation through
`e2a19a2` (base includes v13 `b51af08`). Separate agents performed implementation
and architecture review; neither edited files nor launched GPU jobs.

## Implementation lane

Initial recommendation REQUEST CHANGES:

1. HIGH: implicit namespace `scripts` can lose to an installed package.
   Reproduced by independent reviewer in local conda environment.
2. MEDIUM: environment-driven ROT90/FEATURES missing from manifest.
3. MEDIUM: external overlap exclusion omitted historical held-out geometries.

Fixed in `e2a19a2`:

- Explicit `scripts/__init__.py` and import-origin test.
- Whitelisted experiment environment in manifest, fail closed for ROT90 != 0
  or FEATURES != base.
- All 224 local training chip geometries audited against accepted CEMS areas:
  zero intersections, masks not read. Audit bound to prepared-data hash;
  pretraining rechecks geometry and patch hashes.

Independent recheck: APPROVE, no remaining findings. `py_compile`, CLI import/help,
`make gate`, and `/Users/mc/anaconda3/bin/pytest tests/test_hypothesis_lab.py -q`
passed (14 tests). The same focused tests pass in the isolated k8plus runtime.

## Architecture lane

Status WATCH. No reason to stop the authorized exploratory queue.

- Excluding 35 selection chips from BS confirmation scoring does not eliminate
  adaptive selection: all 144 scored chips previously trained the candidates.
  This is development revalidation, not selection-independent confirmation.
  Neither a positive bootstrap interval nor two seeds proves untouched-test
  superiority; several screened candidates add selection uncertainty.
- AF threshold selection uses calibration only, mining uses fit only. Its
  acceptance remains conditional on chip grouping. Two seeds reuse the same
  split; bootstrap does not refit models or establish event independence.
- External padding is excluded from loss/statistics, but affects convolution
  and BatchNorm. Actual valid fraction min .3263, median .9553 is recorded.
- Pre-only temporal augmentation leaves post/mask unchanged but median NBR
  stability does not prove per-pixel severity invariance. NBR tails and usable
  burn coverage are recorded in input-diagnostics-v1.json.

These limitations are retained in SPEC-36 and the campaign report.

## Synthesis

Implementation APPROVE + architecture WATCH => overall COMMENT.
No merge-ready or independent-test superiority claim is made.
The configured architect profile was unavailable; a separate available agent
completed that lane with the same read-only scope, independently of the author.
