# SPEC-47: SWIR-индексы MIRBI и dNBR2 как входные каналы

Status: active

Requirement: REQ-007

Depends on: SPEC-19, SPEC-43

## Summary

Десять пожаров из 144 честных чипов сети не видят вовсе (SPEC-40…44). Ручной
разбор (`research/blind_panels/blind_fires.jpg`, журнал 2026-09-21) показал, что это
настоящие гари: по dMIRBI (Mid-Infrared Burn Index, 10·B12 − 9.8·B11 + 2) истина
отделяется от фона на всех десяти (0.12–1.15 против −0.05–0.11), тогда как по dNBR
она тонет (0.09–0.28 при фоне до 0.34 у сохнущей травы — литературный предел
обнаружимости в степи). Среди наших 11 оптических каналов **нет B11 вовсе**, нет
MIRBI и dNBR2 — только `nbr2_post`. Ресёч №2 (блок 3b): для низкой биомассы MIRBI
и NBR2 — лучшие индексы, dNBR — худший.

Гипотеза: добавить каналы `mirbi_pre`, `mirbi_post`, `dmirbi`, `nbr2_pre`, `dnbr2`
(и `b11_post`) в `features.stack` и в оптический набор сетей. Проверка в две
ступени: (1) бустинг по пикселю на пяти групповых фолдах с новыми признаками —
CPU, парно к бустингу соседа; (2) оптическая сеть с расширенным набором на пяти
фолдах — GPU, парно к `bs-confirm-optical-f{f}-v1`. Затем 35 чипов.

Честное ожидание: +0.005…+0.015 по пулу, в первую очередь через потерянные
пожары (14 → меньше). Сигнал остановки: пул не выше базы на 0.004, или минус
на 35 чипах (правило двух шкал), или потерянных не меньше. Отдельно записать:
ось «новые спектральные признаки» ранее закрыта по RdNBR и разбросу SWIR на
бустинге на 35 чипах — MIRBI/dNBR2 там не проверялись, а слепота к бледной гари
тогда не была известна.

## Acceptance Criteria

- `src/comp/features.py`: новые имена в `stack`, набор `NAMES_SWIR`; тест на
  форму и на то, что `dmirbi` истины положителен на `BS_tr_000124`.
- Бустинг: `hypothesis_bs.py boost --fold f` с `FEATURES=swir` →
  `research/bs-confirm-boost-f{f}-swir-v1`; `scripts/exp_swir.py` — пул, фолды,
  потерянные, в рецепте v20 с заменой бустинга.
- Оптика: `hypothesis_bs.py train --variant optical` с расширенным набором →
  `research/bs-confirm-optical-f{f}-swir-v1`; тот же скрипт.
- Принято → финальные модели и v21.

## Verification

- `python -m pytest -q tests/test_features.py`; `python scripts/exp_swir.py` на k8plus.
- Не доказывается: закрытая выборка.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
