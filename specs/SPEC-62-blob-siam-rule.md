# SPEC-62: Правило по пятну: принять пятно бустинга по отклику сиама

Status: active

Requirement: REQ-007

Depends on: SPEC-58, SPEC-59

## Summary

Три объектные спеки (SPEC-58/59/60) нашли ровно один признак пятна бустинга с
сигналом — **средний отклик сиама** (AUC 0.71; истинные 0.036, ложные 0.002; у
29 % истинных p > 0.3). Он не использован напрямую: в SPEC-59 стоял вместе со
связностью, которая всё убила, в SPEC-58 — признаком логистики с плохой рабочей
точкой. Здесь — правило без обучения: пятно бустинга (≥ 20 пикс., вне выхода
смеси v22-аналога) принимается со степенью по бустингу, если средняя p сиама > s
и средняя p бустинга > τ.

**Предзаявлено**: s ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5} × τ ∈ {0.5, 0.7}; выбор по
пулу фолдов 0–2; принято, если пул 5 фолдов ≥ +0.004 **и** проверка 3–4 ≥ 0
**и** 35 чипов ≥ 0 **и** потерянных меньше. Ожидание +0.003…+0.008 — та доля
резерва +0.031, которую сиам уже видит. Скрипт `scripts/exp_two_signals.py`.

## Acceptance Criteria

- Таблица s × τ на фолдах 0–2, проверка, пул, 35 чипов, вердикт.
- Принято → правило в `postproc` (нужны карты бустинга и сиама), v23.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/exp_two_signals.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
