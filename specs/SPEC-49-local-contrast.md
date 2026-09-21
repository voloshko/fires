# SPEC-49: Локальный контраст индексов как входные каналы

Status: active

Requirement: REQ-007

Depends on: SPEC-47

## Summary

Оптическая сеть с теми же SWIR-каналами, что подняли бустинг на +0.033, не
сдвинулась (SPEC-47, ступень 2). Ресёч №2 (блок 3c): сеть видит слабую гарь как
отклонение от статистики сцены, бустинг — как абсолютное значение; малое пятно
слабого контраста тонет. Гипотеза: дать сети **локальный контраст** явно —
z-оценку dNBR и dMIRBI в окне 31×31: (x − mean₃₁)/(std₃₁ + ε). Два канала,
хук `LOCAL_Z=1` в `bs_inputs`, пять групповых фолдов парно к оптике соседа.

Ожидание скромное: +0.003…+0.008; сигнал остановки — пул не выше базы на 0.004
или минус на 35 чипах.

## Acceptance Criteria

- `LOCAL_Z=1` добавляет два канала; без переменной поведение прежнее (тест).
- Пять фолдов `research/bs-confirm-optical-f{f}-lz-v1`; замер в `exp_boost_channel.py`.
- Принято → финальные сети и v22/v23.

## Verification

- `python -m pytest -q tests`; `python scripts/exp_boost_channel.py` на k8plus.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
