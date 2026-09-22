# SPEC-68: AF: разбор чипов с концентрацией ошибок и явные признаки I3/I4, глинт

Status: active

Requirement: REQ-007

Depends on: SPEC-38

## Summary

AF даёт 35 % скора при F1 0.952; 18 % ложных и 6 % пропусков сидят на
нескольких чипах, и мы никогда не смотрели, что там. Ступень 1 — диагностика
(CPU): пятикратный OOF по 336 чипам текущим бустингом, порог из
`af_oof_cutoff.json`; концентрация FP/FN по чипам, ночь/день, землепользование
ошибок, спектр ошибок против верных — I1 (глинт: I1 > 0.3 и I1 > I2), I3/I4
(факел: SWIR выше MWIR), сенсорный зенит (край полосы). Ступень 2 — **только если
ступень 1 показала мишень** (≥ 15 % FP под глинт- или факел-подписью, или FP
на краю полосы): явные признаки I3/I4, глинт-флаг, аномалия I4 с поправкой на
зенит — в бустинг, OOF парно. Критерий ступени 2: F1 OOF ≥ +0.002 при не худшем
FP на пустых чипах. Без мишени спека закрывается диагнозом.

## Acceptance Criteria

- `exp_af_badchips.py`: таблицы топ-чипов FP/FN, спектр TP/FP/FN, доли глинт-кандидатов, ночные доли.
- Ступень 2 (условно): признаки в `src/comp/af.py`, OOF-замер.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/exp_af_badchips.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
