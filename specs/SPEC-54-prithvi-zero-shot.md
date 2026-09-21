# SPEC-54: Нулевой выстрел Prithvi-EO BurnScars на степных чипах

Status: planned

Requirement: REQ-008

Depends on: —

## Summary

Первый шаг REQ-008: готовые веса Prithvi-EO-2.0-300M-BurnScars (6 полос HLS:
B2 B3 B4 B8A B11 B12, 30 м, бинарная гарь, IoU 0.875 на их тесте) прогоняются
**без дообучения** на наших 144 групповых чипах (сцена «после», 20 м → 30 м,
нормализация их means/stds). Цель — узнать, видит ли лесная модель степную
гарь вообще. Мерим IoU гари против нашей бинарной истины (степень > 0) на
чистом небе; сравниваем с одиночной оптической сетью соседа (0.618 взвешенно;
IoU гари одиночной сети — посчитать той же функцией). Сигнал остановки REQ-008:
IoU < 0.5. Данные и веса: `~/fires/external/` на k8plus (Apache 2.0, CC-BY-4.0).

## Acceptance Criteria

- `scripts/exp_prithvi_zero.py`: карта вероятностей гари по 144 чипам, IoU по пулу и по фолдам, потерянные чипы, время инференса; `research/prithvi-zero-v1/`.
- В receipt `non_claims`: бинарно, без степени; 30 м против 20 м; сцена «после» без «до».

## Verification

- На k8plus: `.venv/bin/python scripts/exp_prithvi_zero.py`; PASS для продолжения REQ-008 — IoU гари ≥ 0.5 на пуле.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
