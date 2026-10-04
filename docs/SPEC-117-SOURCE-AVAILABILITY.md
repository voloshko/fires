# SPEC-117 CEMS source availability

Current Rapid Mapping listing:265records,2023-03-24 through2026-09-15.
Historical portal listing:1065records,2012-04-27 through2026-09-15.
Both snapshots have complete declared pagination, unique codes and zero
records explicitly tagged Russia/Russian Federation. Historical one unknown
country retained. This rules out a declared Russian training source in these
snapshots, not Russian fires or labels anywhere else.

Country-tagged northern wildfire candidates are Swedish: EMSR660(2023),
EMSR298(2018),EMSR290(2018),EMSR096(2014). Three are2015+ by activation time.
Activation timestamps are not ignition dates. Country membership does not
prove boreal ecology, available vector labels, common Sentinel observations,
unique physical fires, or independent-test admission. No data split/collection
or model training follows from these counts without a separate specification.

Raw snapshots remain onk8plus in research/cems-source-availability-v1 andv2.
Finalv2 replayed1330records exactly;786017bytes before tiny replayproof. Five
CPU tests pass; source recipe42b5d69f3dcbfb624ea2abf34dfbfe5045a30b2cd8ecbeb1d8d43664ef8eb5dd
matches local committed code. Oldv1 immutable. Independent codeAPPROVE after
blank-country preservation; architectureWATCH; finalCOMMENT. PASS receipt
covers metadata only, not model accuracy or original goal completion.

Primary endpoints:
[Recent Rapid Mapping listing](https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations-info/?limit=1000),
[Historical CEMS portal listing](https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=2000&activationTime=2012-01%2C2026-10-04).
The historical endpoint was found in the official portal's published frontend
bundle. This audit is descriptive; initial discoverycounts already observed
before its specification, with no preregistered accuracy-outcome claim.
