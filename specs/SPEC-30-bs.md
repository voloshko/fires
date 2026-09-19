# SPEC-30: Радиометрическая нормализация сцены «после» к сцене «до» для BS

Status: active

Requirement: REQ-007

Depends on: SPEC-19, SPEC-26

## Summary

Спектральное дрожание — имитация другого дня съёмки — дало самую крупную
прибавку среди приёмов обучения (+0.008 на одиночной сети). Значит, разница
условий между сценами «до» и «после» (освещение, дымка, фенология) реально
портит индексы, и сеть учится её терпеть. Классический ход в детекции изменений
— убрать разницу до расчёта индексов: **линейная подгонка каждой полосы «после»
к «до»** по устойчивым пикселям (малое |dNBR|, обе сцены валидны), затем
индексы считаются от нормализованной сцены. Гарь — меньшинство пикселей, и
робастная подгонка по устойчивым её не «замажет».

Точка отсчёта — оптическая пятёрка (SPEC-26): 0.7604 / 0.7255 с фильтром
чужих пожаров, 0.7287 взвешенно без него; лучшая одиночная оптическая сеть с
дрожанием 0.7259 взв. Разброс сида 0.004; всё меньше 0.005 — шум. Стенд
`scripts/exp_unet.py`, замер `scripts/exp_seeds.py`.

**Ожидание:** среднее, +0.003…+0.010 взв. **Сигнал остановки:** две сети с
нормализацией не выше двух без неё (0.7162 / 0.7259) на 0.005 в среднем —
`rejected`. Нормализация — часть признаков, поэтому флаг хранится в бандле
сети и применяется на инференсе автоматически (`unet.probs`).

## Acceptance Criteria

- `src/comp/features.py::normalise_post(chip)` возвращает чип с полосами
  «после», приведёнными к «до» (по каждой полосе `a·post + b` по МНК на
  пикселях с |dNBR| < 0.05 и валидных на обеих сценах; SCL не трогается); тест:
  на синтетическом чипе с `post = 1.2·pre + 100` вне гари нормализация
  возвращает pre с точностью 1 %.
- `exp_unet.py` принимает `RADNORM=1`, бандл несёт `radnorm`, `unet.probs`
  применяет нормализацию, если флаг в бандле.
- Два сида оптика + дрожание + нормализация против двух без неё: строки в
  `evidence/hypotheses.md`, одна шкала.
- Если принято: финальные оптические сети переобучены с флагом, сабмит,
  receipt.

## Verification

- `python -m pytest tests/test_features.py -q -k normalise` зелёный.
- `RADNORM=1 CHANNELS=optical JITTER=0.1 DEPTH=7 WIDTH=32 TAG=d7rn python scripts/exp_unet.py` и строка в журнале.
- Чего не докажет: перенос на закрытую выборку; и что прибавка не от дрожания — контроль есть (те же сети без флага).

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
