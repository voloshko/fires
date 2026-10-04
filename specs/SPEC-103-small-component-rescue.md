# SPEC-103: Small reference-component rescue for paired fusion

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-95, SPEC-102

## Summary

SPEC102 mean improves bothMediterranean IoUs but loses1positiveFLOGAgroup.
Test onefixed small-component safeguard, before looking at component sizes
of thatfailure: retain C1-MM positivecomponents <=64pixels that mean fusion
completelyrejects. This preserves isolatedbaseline evidence while keeping
mean suppression on largefalseareas. Cap64 isfixed (8x8pixelcount heuristic),
not optimized from thatmissed fire. Components are not physicalfireevents.

## Acceptance Criteria

- Mean0.5*(reserve+C1-MM), threshold0.5, samewater/commonvalidmask asSPEC102.
  Label C1-MM binarymask with8-connectedcomponents. For each size1..64,
  ifmeanbinarymask haszero positives insideit, restore the WHOLEcomponent.
  No restoring a subset chosenby GT; no tuningcap/threshold/alphasweep.
  Savedprediction includes rescuepixels and componentcounts asdiagnostics.
- Development only samefrozenRussian77/EMS60/FLOGA81 caches, exactSPEC92parent
  andweights/data binding. Eligibility sameSPEC102:Russia pointgain>=.01,
  EMS/FLOGA pointdelta>=−.01 andlostpositivegroups<=C1-MM. AllCIs/errors saved.
  Ifnoteligible rejectwithoutconsumingcal61. Ifeligible freeze thissingle
  recipe/weights/sources/orderedcalids andthresholds BEFOREcalpredictions.
- Confirmation unchanged: originalcalibration61,>=50windows/20clusters,
  nofit/evalIDs, untouched byearlierregisteredpredictions ormining/statistics.
  ExactC1-MM5/reserve3/Prithvi parents verified. One selectedrecipe only.
- Acceptonlyconfirmation delta>=.01 andCI95lower>0 to BOTH C1-MM andPrithvi;
  reservepointdegradation<=.01, lostpositiveclusters<=C1-MM. Bootstrap2000,
  seed20261003; allsameobservations/water/p0.5/TTA8/BF16. No point-onlywin.
- Masks/schema/hash/shape/nonfinite checks failclosed, append-onlyplan/results.
  Meaningfultests8-connectivity, rejectedvskeptcomponents, sizecap, invalid/
  waternotrestored, staleparents andcompetitoradmission. No trainingchanges.
- All inference onk8plus undercommonflock; canrunbetweenSPEC101jobs. No imagery/
  weights onMac, no deleting priorartifacts. Components/areas arepixelcounts;
  publishmasked_fraction, not unsupportedphysicalfire/groundarea claims.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_component_rescue.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/develop_component_rescue.py --data-root data-store --parent research/russian-transfer-v1-result.json --russian-manifest data-store/russia/manifest.json --base-manifest data-store/forest-pairs/manifest.json --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out research/component-rescue-v1-development.json`.
- k8plus: `bash scripts/queues/queue_component_rescue_confirmation.sh`.
- non_claims: sourceperimeters/GTagreement only, spring2020 annualpairs;
  developmentMediterraneanreuse, no globalrank/fieldtruth; connectedcomponents
  cannot certify physicalfirecount or generalrecall guarantee.

## Resolution

Rejected by frozen developmentguard; calibration61 notconsumed. RussiaIoU
0.351100 versus0.326108 (+0.024992, CI[−0.019341,0.059026]); EMS+0.021228
andFLOGA+0.026363 withpositiveCIs. Yet FLOGA stillmisses1group versus0
forC1-MM. Fixed64pixel rescue cannot repair thisfailure. No cap/threshold
change afterresult, no confirmationGPUrun or productpromotion.
Allregisteredcaches verified andparentcounts reconstructed;3testsPASS.
