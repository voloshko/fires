# SPEC-121: Coupled member probabilities for frozen temporal ensemble

Status: rejected
Requirement: REQ-008
Depends on: SPEC-118, SPEC-119

## Summary

Test one fixed inference-only coupling using immutable118 five-member models.
Current anchor expansion E[p]+(1-E[p])E[q] differs from E[p+(1-p)q] by
Cov(p,q). Additional formula is a hypothesis, not correction of a proven bug.
Pair/post use exactly the same change; no newtraining/seed/threshold selection.
All375 original and69eastern windows are already consumed development.

## Acceptance Criteria

- Authenticate immutable118resultSHAfb6aa7f3fdd8a4f4d24dc4c04e8a9b33ebdd41117927ef24adff5288b907b851,
 119resultSHA5b0d13886122636c4f7bfd88c14b1ed2b4406548a47adcd9b51ac315fe5d3c4b,
 118trainingsealccf1df7dd1eb779f7b2d5d66019751eb675b1b48e916716c3171837692f550dc,
 all10mixednative/model/source/fit identities and originalcomparators/caches.
 Old118/119 states/seals/counts/outcomes/images immutable.
- For eachseed, keepq exactly118 (lockedfloat32 normalization/TTA8probmean/
 BF16), and recompute its C1postp exactly originalreference path (original
 normalizationcasting,TTA8logitmean->softmax/BF16). Uniformdelta=mean5[(1-p_i)q_i].
 Sumrecomputedbasep_i/5 must match authenticatedC1-MM5anchor withinatol1e-6/
 rtol1e-5; otherwisefail beforecache. Candidate=clip(anchor+delta,0,1).
 Clippingonlytiny numerical upperdrift; no negativecorrections/reweighting.
 This equalsmean individuallyexpandedmembers when anchorparity holds. TTA
 probability/logit distinctions remainexplicit; no averaging rawqlogits.
- Retain exactlysamevalid/water/threshold.5, referencepositivepreservation,
 sourcebytes/labels andpopulationorder. Stream per-member accumulation, no
 per-memberfloatcache or additionalbackbonecopies. Pair/postcapacity unchanged.
- Original375development eligibility unchanged (Russian234/90 gain>=.01/
 CIlower>0 vs matchedpost/C1/Prit; regionalpoint>=-.01 toC1/reserve; Medpoint
 >=-.01/lower>=-.03 vsC1; noextra missedgroups). Useful coupling additionally
 requiresRussianpooleddelta>=.005/CIlower>0 vsparent118pair.
- ConsumedEastern69/41 retained asadditionaldevelopment guard, neverfresh:
 same119confirmationcriteria gain>=.01/CIlower>0 vs post/C1/Prit, reserve
 point>=-.01/noextra C1misses. Noclaimindependence or waived119FAIL. If any
 gatefailsreject; acceptedfuturepromotionrequiresseparatelypreregisterednewdata.
- Tests covariance/sign/formula, probabilityrange/referencepreservation,
 exactfivecount, wronganchorparity and losslesspackedcache roundtrip/binding.
 Actual512GPUmechanics bothpair/post plus originalC1reference parity before
 fullinference. Finite outputs; no qualitysmoke claim. Source/inference recipe
 and allsettings fixed before outcome. Do not change sealed118/119helpers.
- Allk8plus, oneGPUlease/free22GB,CPU400/12G. No images/weights downloaded.
 Newlosslesscompressedpackbits caches only, totaloutput<=16MiB andfreefloor3GiB
 beforeeachwrite (remaining rootspace~26MiB abovefloor). Preserve oldcaches;
 no physical rootfilesystemchanges or olddeletion.
- Tests/independentreview/receipts/gate and perspeccommits; coreoriginalfour-goal
 remains unproven, no product/worldrank/physical/severity/L2A acceptance.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_coupled_member_inference.py -q`.
- k8plus: `bash scripts/queues/smoke_coupled_member_inference.sh`.
- k8plus: `bash scripts/queues/queue_coupled_member_inference.sh`.
- `make gate`.
- non_claims: all444windows consumed; mathematical coupling is not guaranteed
 accuracy, independentconfirmation or probability equivalence across different
 TTAorders; no fieldtruth/worldrank/severity/L2Asuccess/promotion/fullgoalcompletion.

## Resolution

Rejected fullinference-only hypothesis, no changedcriteria. All444windows
scored,888compressedmask confusionrows/allCIs/decisions replayed exactly.
Per-population referenceanchorparity and positivepreservation pass. Old375
eligibility remains true, but pooledparent118gain+0.000028621
CI[-0.000358173,0.000308189] is negligible, below.005/nonpositive lower.
ConsumedEastern69 pair0.248125 vsC1.219085/Prit.232015/post.230561;
postCI[-.001245,.036277] andPrithviCI[-.034831,.090360] failunchangedguards.

Scorerterminalexit0/1101.222swall/2.2Gpeak. Totalnewoutputs1072158bytes before
smallfinalaudit, below16MiB; no sampledcohorts/olddeletion. ImmutableFAIL.
Mathematicalcoupling validbutno usefulaccuracy; no source/label/threshold/model
changes or promotion. Originalfour-part goal remains unproven; best118 and
119freshC1gain evidence intact, metadata120pixelrepair stillunexecuted.
