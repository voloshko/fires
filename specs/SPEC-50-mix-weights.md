# SPEC-50: Перенастройка весов смеси после SWIR-бустинга

Status: active

Requirement: REQ-007

Depends on: SPEC-47

## Summary

Веса смеси v21 (бустинг 0.4; внутри сетей оптика 0.5 / сиам 0.5) подбирались под
бустинг из 19 признаков (один: 0.473 на фолдах). После SPEC-47 бустинг вырос до
0.506, а веса не пересматривались. Три спеки подряд (47/48/49) показали, что
входы оптической сети резерва не дают; резерв — в самой смеси. Заодно
пересматривается логит-усреднение сетей (ресёч №2, блок 2), ранее не принятое
по протоколу (+0.004 пул / +0.008 на 35 чипах, проверочные фолды −0.008), — как
одна из осей сетки, а не отдельная гипотеза.

**Предзаявленная сетка**: wb ∈ {0.30, 0.35, 0.40, 0.45, 0.50, 0.55}, доля сиама
s ∈ {0.4, 0.5, 0.6, 0.7}, среднее сетей арифметическое или геометрическое
(бустинг всегда арифметически, `exp_geo_check.py`). Всё остальное как в v21.
Фолды: бустинг SWIR, оптика соседа, сиам — среднее двух передискретизованных
сидов (как в продукте). 35 чипов: пять оптических сетей, один передискр. сиам,
бустинг SWIR (`exp_swir35.py`). Скрипт `scripts/exp_mix_weights.py`, CPU по кэшам.

**Протокол**: комбинация выбирается по пулу фолдов 0–2; принимается, если пул
пяти фолдов выше v21 на > 0.004 **и** проверочные фолды 3–4 не ниже v21 **и**
35 чипов не ниже v21. Сигнал остановки — любое из трёх нарушено; тогда веса v21
остаются, а таблица ложится в леджер. Ожидание честное: +0.002…+0.006.

## Acceptance Criteria

- `scripts/exp_mix_weights.py` печатает таблицу всех 48 комбинаций на обеих шкалах и вердикт по протоколу; JSON в `research/mix_weights_spec50.json`.
- Принято → `inference.py` с новыми весами (`--net-weight`, `--unet-weights`, при геометрии — новый флаг), v22 в `submissions/`, receipt SPEC-19-LIVE-022.
- Отклонено → Resolution с таблицей, строка в `evidence/hypotheses.md`, код продукта не меняется.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/exp_mix_weights.py` — PASS, если строка «КРИТЕРИЙ SPEC-50: ПРОЙДЕН».
- Порог 0.004 и правило минимакса зафиксированы до запуска и не двигаются.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
