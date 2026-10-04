# SPEC-120: Public CDSE metadata audit for L2A protected-pixel losses

Status: active
Requirement: REQ-008
Depends on: SPEC-93, SPEC-97

## Summary

The final PlanetaryComputer L2A manifest loses2784protected positive pixels in
16originalfit windows. Discover public CDSE L2A same-day/same-MGRS descriptors
for precisely those windows before any credential-dependent pixel repair.
Metadata availability is not visibility, preserved labels or source admission.

## Acceptance Criteria

- Exact final manifestSHA75d1afa37b8dd51d98ea23c2d364b6f1f24e6c253e8ed354f7167d2dcfe3794b,
  SPEC93/qc_probefalse/readyfalse, original886known/retained861,16losswindows/
 2784pixels. Use its frozen full lost_protected list, preserve all IDs/counts.
- Derive acquisition calendar day/MGRS strictly from frozen HLS sample names;
  query public catalogue.dataspace.copernicus.eu/odata/v1/Products with Sentinel2
  collection/L2A/sameMGRS/sameUTCday. No guessed fire dates, relaxed dates or
  different tiles. Capture raw response/request time/status/hash/URL.
- Complete count/unique product/pagination validation; malformed/HTTP failure
  gets unknown/error, not zero/absence. Keep offline/missingOnline separately.
  Parse product sensingday/tile/baseline, verify requestedidentity and ContentDate.
  Report online/offline/unknown descriptors. Do not pick by label/model outcome.
- All16results retained, including complete-empty listings and query failures.
  Unknown processing baselines reported; no radiometric/asset validity assumed
  from Name alone. No token/header/credential read or prints, OAuth, downloads,
  band/vector/scene imagery or GPU/model inference; public metadata only.
- Own k8plus research/cdse-retention-metadata-v1; serialrequests/OMP2/CPU200/4G,
  totaloutput<=8MiB/freefloor3GiB before everywrite. Original manifests/proofs/
  images/masks/readiness untouched. Testsidentity, unknown/offline, malformed/
  partial/list duplicates and lostpopulation validation. Source receipt/report,
  independentreview/gate/perspeccommits.
- Keep93/97readyfalse regardless descriptors. Actualpixelrepair requires separate
  spec/authorizedsourceconfiguration, exactlysame heldwindow identities/supervision,
  strict original SCL and allprotectedpixel guarantee before training.

## Verification

- k8plus: `.runtime/bin/python -m pytest tests/test_cdse_retention_metadata.py -q`.
- k8plus: `.runtime/bin/python scripts/audit_cdse_retention_metadata.py --manifest data-store/l2a-pc-v1/manifest.json --out research/cdse-retention-metadata-v2`.
- `make gate`.
- Primary reference: [CDSE OData documentation](https://documentation.dataspace.copernicus.eu/APIs/OData.html).
- non_claims: metadata/Online flag not pixels/cloudfree coverage/labelretention,
  no fieldtruth/fireabsence/L2Ausefulgain/competitorimprovement/promotion or
  completion of originalfour-directiongoal.

## Execution notes

V1 completed16completequeries/16online descriptor windows. Independent
architecture review BLOCK: Requestsdefaulttrust_env permits .netrc/environment
authentication, contrary to promised no credential reads. No actual access or
exposure established. ImmutableFAIL/sourceversion/firstrawresult preserved.
ExplicitSessiontrust_envFalse and regression prevent environment authentication
andproxylookup. Five tests pass; finalv2 uses a new immutable output/source seal.
No pixel/sourceadmission claim from either metadata snapshot.
