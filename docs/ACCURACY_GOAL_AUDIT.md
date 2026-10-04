# Original four-direction goal: current completion audit

This audit keeps the original scope. None of the following rows is a claim
that the full user goal is complete. All research outputs and negative results
remain preserved; no experimental model has been promoted to the product.

| Direction | Authoritative evidence | Current conclusion |
|---|---|---|
| Learned two-date forest/mountain detector | SPEC-118-DEVELOPMENT-001; SPEC-119-CONFIRMATION-001 | Ten-member experiment passes consumed-development criteria. Fresh geography confirms gain over C1-MM, but matched-post and Prithvi intervals cross zero. Full independent superiority remains unproven. |
| Russian/Siberian labels and hard negatives, new fires isolated beforehand | SPEC-92; immutable Russian447 and prior-footprint registries; SPEC-112 fixed selection; SPEC-119 source/model binding | Russian309fit labels used and new69/41 geography excluded from prior fit before predictions. This geography is now consumed. Improvement over C1-MM is confirmed there; Prithvi superiority and broad Siberian transfer are not established. |
| Direct L2A fine-tuning | SPEC-93 active; SPEC-97 retention bound; SPEC-101-REPLAY-001; SPEC-120-METADATA-001 | Real six-run full929-supervision mixed replay was implemented, but useful gain failed. L2A-only original-event coverage remains unproved:2784protected pixels lost in16windows. CDSE has online descriptors for all16, with no pixel repair yet. |
| Paired severity within fixed detected contour, balanced classes | SPEC-94-REPLAY-001; SPEC-100-REPLAY-001; SPEC-110 legend audit | Balanced head improves hybrid but loses to equal-weight control. Frozen equal head on173fresh events improves point mIoU but CI crosses zero. MTBS remains satellite-derived agreement, not field severity. |

## Numbers that must not be mixed

- SPEC118 Russian234windows/90groups: pairIoU0.287230, C1-MM0.219247,
  Prithvi0.116043, matchedpost0.228190. All fixed eligibility guards pass.
  These are consumed development, not competition hidden-test/world ranking.
- SPEC119 previously unused geography69windows/41groups: pair0.247597,
  C1-MM0.219085, Prithvi0.232015, post0.230528. C1delta+0.028512
  CI[0.010246,0.042939]; PrithviCI[-0.035064,0.089592] and postCI
  [-0.001678,0.035502] inconclusive. Source maskedfraction0.151336.
- SPEC101 L2A replay0.674091 versus originalC1.673324, delta+0.000766,
  CI[-0.003273,0.004937]; no useful-gain acceptance.
- SPEC94 balancedseverity0.363802/equal0.367197/hybrid0.308888. SPEC100
  fresh173 equal0.433595/hybrid0.407446, delta+0.026149,
  CI[-0.007822,0.057260]; no independent acceptance.
- SPEC121 inference coupling changes oldRussianIoU by only+0.000028621,
  CI[-0.000358,0.000308]; rejected. All444windows remain consumed.

## Required next evidence and pending inputs

1. A new preregistered population is required for another independent claim.
   Do not treat SPEC119's69windows as fresh again or waive its failure.
2. k8plus root currently has about33.6MiB above the enforced3GiB floor,
   insufficient for another meaningful image population. Read-only filesystem
   checks found5%root reserve,6237555blocks×4096bytes. A proposed2%reserve
   would release about14.3GiB while retaining9.5GiBroot reserve. Explicit user
   approval is pending; no filesystem setting or original data has changed.
3. Optional human request for an existing CDSE configuration path onk8plus
   remains unanswered. No credentials or tokens supplied/read. Pixel repair
   needs a separate spec and strict same-date/MGRS/grid/SCL/protected-pixel
   checks; online metadata alone cannot set SPEC93/97 ready.
4. Further severity work must address the actual grade errors and then pass
   independently on preregistered events. Do not hide MTBS raw0/unassessed
   pixels, convert unknown background into reviewed negatives, or change the
   primary metric to obtain a PASS.

## Workspace and execution

- Own worktree `/Users/mc/.codex/worktrees/fire-accuracy-next/fires`, branch
  `codex/fire-accuracy-next`; original `/Users/mc/projects/fires` untouched.
- Compute/data `/home/mc/fires-accuracy-next` on `ssh k8plus` only.
- Own latest train/source/score units for118/119/121 are terminal exit0.
  There is no current own GPU computation to restart or poll as live.
- Preserve all models, inputs, partial source files, caches, source seals and
  receipts. Future GPU work uses `research/accuracy-gpu.lock`,22GB free VRAM
  admission and its declared memory/CPU/storage limits.
- Current next actions require the pending capacity/access decisions or a
  separately specified, justified experiment on existing consumed data. No
  automatic permission from elapsed time or a default question selection.

## Blocking audit after completed experiments

The same capacity/access condition persisted across three consecutive goal
turns following the final121 run: available headroom above3GiB floor only
25,096,192bytes at currentverification; no liveown GPUjobs, no newunused image
population admitted, no CDSEcredential configuration or rootreserveapproval
received. Existing independent populations have been consumed. More experiments
on these cannot establish the requested new independent superiority by themselves.

No protected inputs/outputs have been removed, rootreserve is unchanged, and
credentials are not inferred from missing replies or defaultquestion options.
Goal status is blocked pending an access/capacity decision, not complete or
user-requested paused. Original four-direction scope and all fixedcriteria remain.
