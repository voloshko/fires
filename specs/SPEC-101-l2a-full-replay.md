# SPEC-101: L2A adaptation with complete original supervision replay

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-93, SPEC-95, SPEC-96, SPEC-97

## Summary

SPEC93 has1008PCtwins but unknown/residualevent-retention proof fails; no L2A
training occurred. Test a different full-pool adaptation recipe: every original
sample supplies alloriginalvalidHLS supervision on odd epochs; on even epochs
candidate receives strictL2Atwin while matchedcontrol receivesHLS with the SAME
commonvalidmask. Missingtwin fallsbackto originalHLS in both. This keeps the
full929originalfitpool and tests direct L2Aweightupdates without inventing IDs,
removing samples, unmasking clouds, or claiming SPEC93/97 passed.

## Acceptance Criteria

- Freeze source_jobs SPEC93 fullfit929: HLS540/MTBS300/EMS49/FLOGA40, hashes of
  sourceimages/GT, originalvalidpixels/positivepixels, grids/bboxes/weightsMed×2.
  No sample removed. PC l2a-pc-v1 twins1008 are a source-bound bank, not an
  admissionPASS; ready_for_training=false is preserved. Source/label/grid/date/
  MGRS/scale/B8A/SCL compatibility checked for every usedtwin. No EarthSearch.
  Record actual889fit twins, allfallbackIDs and missing/QC reasons beforetraining.
- Freeze originalfit and availablePCevaluation frames. Remove only evaluation
  frames intersecting fulloriginalfit before predictions, retaining exclusions
  and bothEMS/FLOGA. No fitdropping. Require>=50evaluationwindows and>=20groups,
  >=10groups perEMS/FLOGA and>=300realfitL2Atwins. OtherwiseINSUFFICIENT_EVIDENCE.
  Newstandalonemanifest authorizes thisrecipe, never changes SPEC93 readiness.
- Seed1/2/3 initializers are exactC1-MM rawweightsSHA from committedSPEC92result
  SHA242acac3cf2ccd30728e5919997895e7cf1d83de223a8802b06f3c4daaaa296d.
  Sixband width32depth7; use eachinitializer's own fit-only mean/std unchanged
  in candidate/control. Eval/currentoutcomes do not determine normalization.
- 30epochs/batch8/AdamWlr3e-5/cosine/BF16/C1augmentation/CE[.25,1]+Dice.
  Everyepoch processes a permutation of everyoriginalsample (Med×2); last
  incompletebatch retained, no replacement sampling/drop_last. Sameindices,
  optimizationsteps/augmentationdraws/seeds in matchedcontrol andcandidate.
- Oddepochs: both originalHLS+fulloriginalvalidGT. Evenepochs withtwin:
  controlHLS andcandidateL2A, identicalcommonvalidmask, GTignore=-1 and identical
  unknownpixel fill BEFORE normalization. Missingtwin: both originalHLS.
  Thus eachoriginalsample has>=15 fullHLS exposures; actual source/GTpixel/
  sampleexposures logged. Unknown IDs remainunknown; no70%L2A-onlyeventclaim.
- Frozen evalPC L2A inputs/masks to candidate3/control3/originalC1-MM5,
  TTA8/threshold0.5/commonwatermask. Same accuracycriteria asSPEC93:
  pooleddelta>=.01 and eventbootstrapCI95lower>0 to BOTH matchedcontrol and
  originalC1-MM; everyEMS/FLOGA delta>=−.01 and lower>=−.03 to both.
  Bootstrap2000 seed20261003. Perwindowconfusions/misses/counts persisted.
- Source/input/recipe/tensor-boundmetadata/lineage checked beforeeveryjob and
  scoring. CPU/preparation/GPU onlyk8plus; no imagery/weights onMac.
  OwnGPUqueue locks perjob and unlocks betweenjobs soSPEC100score canrun.
  Preserve allpreviousmodels/data, nativeweightsavoidduplicateexports.
- Meaningfultests source/mask matching, completeeveryepoch/lastbatch, fallback,
  oddexposure guarantee, staleinputs/initializer/profile exclusions. Synthetic
  smoke is not accuracy; no acceptance from trainingloss or partialseeds.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_l2a_full_replay.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/prepare_l2a_full_replay.py --data-root data-store --event-registry data-store/hls-events.json --twins data-store/l2a-pc-v1/manifest.json --reference-root /home/mc/fires/research --reference-result research/russian-transfer-v1-result.json --out data-store/l2a-full-replay-v1`.
- k8plus: `bash scripts/queues/queue_l2a_full_replay.sh` (all6models thencomparison).
- non_claims: mixedsource adaptation, not SPEC93's L2A-only eventcoverage proof;
  published satellite-label agreement, not fieldtruth; development Mediterranean
  checks, not independent Siberian/globalcompetitor ranking. Fullgoal persists.

## Resolution

Rejected afterall6full30epochruns andmatchedsource comparison119windows/116groups.
ReplayIoU0.674091, HLScontinuation0.671949 (+0.002142 CI[−0.002164,0.007027]),
initialC1-MM0.673324 (+0.000766 CI[−0.003273,0.004937]). Bothprimary delta>=.01
andCIlower>0 fail; no thresholdchange, productnotpromoted. FLOGA gains.017521
overcontinuation butonly.003593 tooriginal (inconclusive); EMSnearunchanged.
All929originalfitrecords retained, minimum15fulloriginalHLS exposures/sample
perseed, matchedorder/common masks in3pairs,14250actualL2Aviews/replayseed.
This proves realdirectL2Aweightupdates/fullpoolreplay execution, notbeneficial
accuracy orSPEC93/97 L2A-onlyeventcoverage. Fullsource/model/normalization
lineage, confusions/errors/checkpoints andFAILreceipt retained.
