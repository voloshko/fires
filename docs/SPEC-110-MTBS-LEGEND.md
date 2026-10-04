# SPEC-110 — Meaning and contribution of MTBS codes

The official [MTBS FAQ](https://www.mtbs.gov/faqs) distinguishes outside-
assessment/background0, unburned-to-low1, low2, moderate3, high4, increased
post-fire greenness5 and non-mapping6. Code0 does not establish independently
reviewed absence of fire; code5 is a burned response, not an unburned site.
Code1 is a mixture of unburned and small/low effects. These distinctions matter
when interpreting the existing convention that maps0/1/5 to benchmark class0.
The [product description](https://www.mtbs.gov/product-descriptions) identifies
these as thematic satellite products, not field measurements.

Our source exporter [build_mtbs_sev.py](../scripts/build_mtbs_sev.py) requests
an annual MTBS ImageServer layer as U8 with nearest-neighbour interpolation,
then writes its values directly. The cached .sev.tif arrays retain0–6; there is
no remapping that would give our raw0 or5 different semantic meanings.

The audit reconstructs every legacy4x4 confusion from signed model/contour
caches:172 development windows and173 previously evaluated covered windows.
No model inference, new imagery, class/threshold/mask/contour modification or
new acceptance calculation occurred. Results and old FAILs are preserved.

| Quantity | Development172 | Covered173 |
|---|---:|---:|
| Observed pixels | 43,858,159 | 44,484,013 |
| Masked fraction (spectral observation) | 0.027293 | 0.019115 |
| Observed raw0 outside assessment | 40,739,949 | 41,532,336 |
| Observed raw5 increased greenness | 4,436 | 5,857 |
| Equal-head legacy background grade FP | 455,631 | 416,115 |
| Of those, on raw0 | 314,833 (69.10%) | 200,280 (48.13%) |

These are legacy benchmark penalties, not counts of proven physical false
fires. The raw0 contribution is identical for all grade heads because they
share a fixed C1-MM contour and force a positive degree within it. Altering
the grade head alone cannot recover low-severity pixels missed by that contour.

For raw2 low severity in development, the contour misses1,121,503 observed
pixels. Among the1,131,148 reached by the contour, equal-head grade errors
are41,255 versus206,029 for the hybrid. For raw3 moderate severity, equal has
85,188 grade errors on reached pixels versus59,607 for the hybrid. On the
covered data these quantities are98,602 versus219,068 for low, and221,944
versus191,860 for moderate. Thus the neural head helps low severity while
moderate remains a specific weakness; boundary coverage is a separate limit.

Do not replace the full-frame benchmark with an easier mapped-only score,
relabel a FAIL as PASS, infer ignition dates or train raw5 as physical no-fire.
Any future source-aware grading must distinguish detection contours, degree,
ungraded/increased-greenness response and evaluation coverage explicitly.
This audit establishes interpretation and exact decomposition, not improved
model accuracy, independent confirmation, field severity or goal completion.
