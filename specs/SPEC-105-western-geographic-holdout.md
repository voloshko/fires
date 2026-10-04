# SPEC-105: New geographic holdout for frozen Siberian paired specialist

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-95, SPEC-96, SPEC-104

## Summary

All current recipes completed their measurements without full acceptance.
A frozen paired-reserve comparator scored0.304575 versusC1-MM0.205915 and
Prithvi0.142659 on alreadyreservedcal61; directsecondary CIs are positive.
Selectionafterthismeasurement needs newgeography, not anotherfusion/cap sweep.
Metadata-only PANGAEA audit65–80E/54–65N finds10143validcontours,7635>=1ha.
Validate unchanged pairedweights on geographicallydisjoint WesternSiberia.
This doesnot close oldFAILs, severity, L2A usefulgain orprospective summerfires.

## Acceptance Criteria

- Beforeimagery/inference, freeze sourceZIP/metadata hashes and Westernbbox
 65–80E/54–65N, valid polygons>=1ha,20km connectedspatialgroups (EPSG3576 asoriginal92; areaalsoequal-area). ExcludeWHOLE
 newgroups intersecting/buffer-near anyprior footprint or previousRussian
 fit/cal/eval frame; priorregistry andallold447frames required; addnew173SPEC100frames. Buffer20km inmetricCRS, preserveallvalidsourcepolygons forgrouping/rasterization, evenbelow1ha. No identities
 from GPKG reorderedrings; rawWKB IDs. No selection from modelpredictions.
- Frozen order SHA256('spec105:'+geometry_id); roundrobin sourceareas1–25,
 25–250,>250ha; cap96frames, max4pergroup, distributeacrossgroups first.
 Frame512x51230m UTM, original sourceperimeters/allinwindow labelled asbefore.
 Min50admittedwindows/20groups orINSUFFICIENT_EVIDENCE. Bothmodels onidentical
 commonvalidpixels/watermask,not weaker cropped/missing-data comparisons.
- Same annualsourceinterval Jan1–May15 2020, seasonalpreMay16–July15 2019,
 postMay16–July15 2020, sameMGRS/grid; PCprimary only. StrictSCL,pre/postbad<=.3,
 commonvalid>=.5 and>=9labelledfirepixels. No fakeignitiondates/reviewednegatives.
- Newoutputs onlyownk8plus data-store/western-holdout-v1. No oldassetsdeleted.
 Predeclare totalnewdata budget<=1.5GiB andminimumfree3GiB beforeeachwrite,
 include partialdownloads/retainedfailures, workers2. Stopbeforebudgetoverflow;
 insufficientadmission cannot lower50/20 or change3GiB/budgetafteroutcome.
 Download onlyrequested30m windows, never fulltilearchives. No images/weightsonMac.
- Freeze SPEC92reserve3/C1-MM5/Prithvi hashes/config andsource/preprocessing
 beforeinference; no newtraining ornormalization. Maincandidate pairedreserve.
 Candidate must beatbothcomparators delta>=.01/CI95lower>0; noadditionalmissed
 positivegroups versusC1-MM. Bootstrap2000seed20261003; allconfusions/FP/masked
 fraction andlimits. Minimumdata thresholds andgeometry fixedbeforepredictions.
- Ifitpasses, a SEPARATEspec isneeded for boundedregionalproductrouting;
 outsidevalidateddomain originalC1-MM mustremain. No directpromotion bymetadata,
 no globalrank/groundtruth/season independence fromsinglepublisher spring2020.
- Test whole-grouppurge/stalehash/budgetfailclosed/identity/samegrid/interval;
 makegate; append-only plan, data manifest, resultandreceipts.

## Verification

- ImplementCPU metadata/admission first; do notstartinference until frozen
 plan andfull50/20dataadmission pass. Allcompute/data sshk8plus.
- k8plus: `.runtime/bin/python scripts/prepare_western_holdout.py --data-root data-store --out data-store/western-holdout-v1 --plan-only` (freeze/checkeligiblepopulation first).
- k8plus: `.runtime/bin/python scripts/prepare_western_holdout.py --data-root data-store --out data-store/western-holdout-v1` (sameplan; sourcecollection).
- k8plus: `.runtime/bin/python scripts/score_western_holdout.py --data-root data-store --manifest data-store/western-holdout-v1/manifest.json --runs research/russian-transfer-v1 --reference-root /home/mc/fires/research --prithvi-root /home/mc/fires/external/prithvi-eo2-300m-burnscars --parent research/russian-transfer-v1-result.json --out research/western-holdout-v1-result.json`.
- `python -m pytest tests/test_western_holdout.py -q`; gatePASS.
- non_claims: independentgeography only, sharedPANGAEAannotationmethod/season;
 satelliteinterpretation, notfieldtruth; no summerSiberian/grade/L2A acceptance.
