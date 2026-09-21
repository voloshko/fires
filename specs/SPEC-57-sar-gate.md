# SPEC-57: Радар под маской облаков как гейтированный вход сиама

Status: active

Requirement: REQ-007

Depends on: SPEC-43, SPEC-47

## Summary

Под маской облаков (SCL любой даты) IoU гари 0.61 против 0.76 на чистом небе;
там решают сети, и это четверть всех ошибок (бриф №2, блок 5). Радар Sentinel-1
как глобальный признак бустинга дал ноль (бриф №1). Ресёч №3 (блок D): в
литературе по наводнениям и гарям радар полезен **только там, где оптика
закрыта**, а как безусловный вход он размывается на чистом небе, где оптика
сильнее (Tanase et al. 2020; Punjab 2025: +21 % площади под облаками).

Гипотеза: сиамской сети подаются четыре канала — VV и VH «после», ΔVV и ΔVH —
**умноженные на маску облаков** (1 под SCL-invalid любой даты, 0 на чистом небе).
Хук `SAR_GATE=1` в `extra_channels`, применяется теперь и к сиаму;
`SiameseUNet` принимает число вспомогательных каналов из данных (11 + 4), бандл
хранит `in_channels`, `unet.load` читает его. Сиам с передискретизацией, сиды
20260930+f, пять групповых фолдов парно к `…-over-v1`, замер `exp_siam_diff.py`
с новой колонкой «под маской».

Ожидание: +0.003…+0.008 по пулу за счёт маски при неизменном чистом небе.
**Сигнал остановки**: пул не выше базы на 0.004, **или** IoU под маской не выше
0.61 у базы, **или** чистое небо ниже базы более чем на 0.002, или после первого
фолда −0.02. Принято → скрининг 35 чипов, финальные сети, v22. Что не делается:
радар в оптическую сеть (нет пары дат — нет Δ; ось входов оптики закрыта),
обучаемый гейт по SCL (маска задана явно, обучать нечего при 144 чипах).

## Acceptance Criteria

- `SAR_GATE=1` даёт 4 канала, нулевых на чистом небе (тест); сиам принимает 33 канала (тест).
- Пять фолдов `research/bs-confirm-siam-f{f}-sar-v1`; таблица `exp_siam_diff.py` с колонкой «под маской».

## Verification

- `python -m pytest -q tests/test_extra_channels.py tests/test_siam_fusion.py`; на k8plus `.venv/bin/python scripts/exp_siam_diff.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
