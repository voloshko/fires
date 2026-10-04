# SPEC-115 source audit result

Final v3 audited all540 original HLS fit images/labels and the17 unresolved
SPEC96 records, with exact registry/vector/publisher bytes and source seal.
It completed exit0 in51.479seconds,4.3GiB host peak,381.9MiB swap peak.
Four tests pass. V1 software failure is preserved; v2 successful result is
also preserved. V3 has identical diagnostics to v2 under repaired date handling.

The two publisher indices have filename-only image/mask columns. Unresolved
rasters contain AREA_OR_POINT=Area and no band descriptions or event IDs.
No exact label/domain/grid reuse was found among540 fit scenes. This strict
absence does not establish independently created masks under other visibility.

Three unresolved scenes have older contours with>=95% geometric overlap:
T16TGQ2019158 (2014 contour), T17SMV2021096 (2016), T17SPS2019094 (2017).
Their more recent candidates also overlap nearly all labels. Repeated prescribed
burn geography is a plausible explanation of ambiguity, not established source
attribution or physical dates. Three scenes have no prior-ignition candidate
in this vector snapshot: T14SPJ2018125,T14SQG2018125,T15SUC2018127.
No candidate means unresolved source association, not no fire.

Every output event_assignment remains null. None of the17 receives a publisher
crosswalk. Source attribution for SPEC93/96/97 remains insufficient; original
450day association rules, protected-pixel guarantees and cloud masks unchanged.
This audit is implemented evidence, not an accepted model or an L2A-ready claim.

Independent code review APPROVE, architecture CLEAR; empty-schema/date repairs
received a follow-up code APPROVE. Final scoped review APPROVE for source audit.
It does not grant event retention, accuracy or product promotion.

Primary publisher documentation:
[HLS Burn Scars dataset card](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars/blob/54f428f7e1839456d21f0eea80c9c723dbd8fb56/README.md).
The card identifies MTBS-derived masks and centered/offset chips, but does not
provide a per-file event table. The locally authenticated indices are the
direct evidence for their filename-only schema.
