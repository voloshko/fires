# SPEC-104: Reference-preserving high-confidence paired additions

Status: rejected
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-95, SPEC-102, SPEC-103

## Summary

SPEC102mean improvesIoU butloses fiveC1-MMtruepixels inFLOGA786775.
SPEC103componentrescue cannotrepair it (106pixelcomponent alreadypartlyretained).
Test one monotonic recipe: keepALL C1-MMp>=.5 detections and onlyadd paired
p>=.95. Unlike maximumat.5, .95 prioritizesconfident additions tocontrolFP.
The addedthreshold isfixed BEFORE thisrecipe's results, no sweep/tuning.

## Acceptance Criteria

- Booleanprediction = ((C1-MM>=.5) OR (reserve>=.95)) ANDcommonvalid ANDnotwater.
  Referencepositivepixels cannot be removed; provewithtest. This is a pixel
  recallproperty onidenticalobservations, not a physicalfire/groundtruthclaim.
  No source-mask/threshold/normalization changes to eitherexistingnetwork.
- SamefrozenSPEC92result/weights/sources andverifieddevelopmentcaches as102/103.
  SameRussian77/EMS60/FLOGA81 populations. Eligibility:RussianIoUgain>=.01,
  everyMedpointdelta>=−.01, noadditionalmissedgroups. PublishCIs/errors/addedTPFP.
  Ifnoteligible rejectwithoutcal61; do notmove.95afterresult.
- Ifeligible, append-onlyplan beforecalpredictions stores fixed.5/.95 rule,
  ordered originalcalibration61 IDs, allinput/model/source hashes. Sameunused
  populationmin50windows/20clusters, nofit/evalIDs orregisteredpriorpredictions.
- Confirmationacceptance:delta>=.01 andCI95lower>0 toBOTH C1-MM ANDPrithvi,
  reservepointdegradation<=.01, noadditionalmissedclusters. Bootstrap2000seed20261003,
  p/TTA8/BF16/water/commonmasks andallparentlineage as103. Preserveoriginalsplit.
- CPUdevelopment/newGPUconfirmation onlyk8plus, serializedflockbetween101jobs.
  No image/model downloadlocally, no olddata/weight deletion. Masked_fraction,
  perwindowcounts andhonestlimits mandatory; no promotionfromdevelopment alone.
- Testsstrict.95boundary, referencepreservation, validity/water/shape/nonfinite
  andbothcompetitorrequirements. No collateraleditto sealedSPEC101trainingcode.

## Verification

- `ssh k8plus 'cd /home/mc/fires-accuracy-next && .runtime/bin/python -m pytest tests/test_reference_additions.py -q'`; `make gate`.
- k8plus: `.runtime/bin/python scripts/develop_reference_additions.py --data-root data-store --parent research/russian-transfer-v1-result.json --russian-manifest data-store/russia/manifest.json --base-manifest data-store/forest-pairs/manifest.json --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --out research/reference-additions-v1-development.json`.
- non_claims:previouslyevaluateddevelopment, sourceGTagreementnotfieldtruth,
  Russianannualspring2020pairing; components/clustersnotphysicalfireIDs,
  no globalranking/prospectiveoruncomparedmodelclaims.

## Resolution

Rejected by originalno-degradation-to-reserve safeguard. Fixed61window/29cluster
confirmation completedexit0, calibrationpopulation nowconsumed. FusionIoU0.231959,
C1-MM0.205915 (+0.026044 CI[0.006281,0.043321]), Prithvi0.142659
(+0.089299 CI[0.035291,0.207044]): bothnamedcompetitors beatenhere.
But reservealone0.304575, fusiondelta−0.072616 CI[−0.127648,−0.017177],
belowallowed−0.01. Cannotwaive thisfrozen criterion orpromote onpartialwins.
Perwindowconfusions/rescuecounts/lineage/recipe preserved; no GT/thresholdedit.
No longer claimcal61unused innewexperiments; anyreuse isdevelopment.
Regionalpositive signal exists butnotglobal/fullrecipeacceptance.
