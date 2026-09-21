# SPEC-55: Стенд трёх биомов: степь, лес, горы

Status: planned

Requirement: REQ-008

Depends on: SPEC-54

## Summary

Стенд трёх биомов — общая линейка для всего REQ-008. Степь: наши 144 чипа по
пяти групповым фолдам (метрика REQ-007 плюс IoU гари бинарно). Лес: HLS Burn
Scars, тестовый сплит 264 сцены, IoU гари. Горы: FLOGA (Греция, пары до/после,
экспертная разметка), IoU гари. Один скрипт печатает три числа для любой модели,
умеющей выдать вероятность гари по 6 полосам. Стенд фиксируется до первого
дообучения и не меняется после — иначе сравнения теряют смысл.

## Acceptance Criteria

- `scripts/biome_bench.py <predictor>` печатает таблицу биом × метрика; манифест данных с хешами в `research/biome-bench-v1/`.
- Загрузка FLOGA — отдельно, с лицензией в манифесте.

## Verification

- Прогон стенда на v21-рецепте (степь) и на Prithvi (все три) даёт первую строку таблицы; числа — в receipt SPEC-55.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
