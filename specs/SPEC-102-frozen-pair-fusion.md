# SPEC-102: Frozen post and paired fusion with unused Russian confirmation

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-91, SPEC-92, SPEC-95

## Summary

SPEC92 paired reserve beatsPrithvi butnotC1-MM andfailsFLOGA. Predictions may
contain complementary truepositives. Test twofixedfusions cheaply from completed
caches, selectby recordeddevelopmentrule, then confirm one frozenrecipe on the
61previously unused Russiancalibrationwindows. Thosewindows are consumedas a
singleconfirmation set, not usedfor tuning. No newtraining/alphasweep.

## Acceptance Criteria

- Exact SPEC92 parentresult SHA
  242acac3cf2ccd30728e5919997895e7cf1d83de223a8802b06f3c4daaaa296d,
  reserve3 andC1-MM5 rawweight hashes; Prithvi referenceweight/config from same
  completedrun. Freeze selectionplan/model/source/data hashes beforecal predictions.
- Candidates only arithmeticmean0.5*(reserve+C1-MM) and elementwisemaximum;
  p0.5/TTA8/BF16, identicalcommonvalid/watermask. No clipping/thresholdchange.
  Development uses existingRussian77 andEMS/FLOGA60/81 caches, never freshcal61.
  Signed caches andparent counts reconstruct exactly; no inference for missing
  developmentcache. Publishallcandidates regardlessofoutcome.
- Developmenteligibility: RussianIoU gain>=.01 toC1-MM; EMS/FLOGA pointdelta
  >=−.01 and numberofpositive groups completelymissed<=C1-MM ineachcohort.
  CI reported butdevelopmenteligibility is notpromotion. Ifnoneeligible,
  reject withoutconsumingconfirmation. Ifseveral, greatestRussianIoU wins,
  stabletie alphabeticalname. Do notselectbasedon confirmation.
- Confirmation only originalpartitioncalibration61 fromimmutable447window
  Russianmanifest, >=50windows/20spatialclusters, no originalfit/eval IDs.
  Allmodel normalization/miningfit IDs disjoint, preserveoriginalpartition.
  Verify no pre-existing modelprediction caches for thispopulation. Unexpected
  caches block the unusedclaim ratherthan silently continuing.
- Beforeconfirmation, append-only plan stores chosenrecipe, orderedids, input/
  weight/source/parent hashes andguardthresholds. Onefixedselectedrecipe.
  Acceptedonly ifconfirm delta>=.01 andCI95lower>0 toC1-MM ANDPrithvi; notbelow
  reserve by>.01, no moremissedpositiveclusters thanC1-MM. Bootstrap2000,
  seed20261003. Always reportcomparisons/perwindowcounts/masked_fraction.
- Russiancluster != physicalfireID; missedclusters/windows are labelledas such.
  Dataset spring2020 annualpairs/publishedperimeters, notfieldtruth; development
  Medresults do notprove freshMediterranean superiority/globalranking.
- Small meaningfultests fusion preservesmax-recall, mean/shape/nonfinite and
  development selection rules, stalehash/source/confirmationadmission.
  SameGPUflock perstandaloneconfirmationjob, canrun betweenSPEC101trainingjobs.
  Code/evidence onlylocally; all modelinference/data k8plus.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_pair_fusion.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/develop_pair_fusion.py --data-root data-store --parent research/russian-transfer-v1-result.json --russian-manifest data-store/russia/manifest.json --base-manifest data-store/forest-pairs/manifest.json --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out research/pair-fusion-v1-development.json`.
- k8plus: `bash scripts/queues/queue_pair_fusion_confirmation.sh`.
- Gate/review/synthetictests alone do notprove improvement. No winnerfrom
  pointestimate withoutconfirmationCI andallnamedcompetitors.

## Resolution

Rejected beforeconfirmation by frozeneligibilityrule; calibration61untouched.
Mean:Russia0.350121 versusC1-MM0.326108 (+0.024013, CI[−0.019097,0.059354]);
EMS+0.021424 andFLOGA+0.027877 with positiveCIs, but FLOGA missedgroups1>0.
Maximum:Russia0.356746 (+0.030638), butFLOGA−0.055482 violates−0.01.
Neithereligible; selected_recipe=null. Reportboth, no promotion/thresholdchange.
CPU-only cachedprediction checks passed, parentcounts reproducedexactly.
Do notconsume confirmation just to rescue a developmentfailure.
