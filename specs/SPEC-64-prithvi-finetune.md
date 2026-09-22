# SPEC-64: Дообучение кодировщика Prithvi-EO-2.0 на степных чипах как член смеси

Status: active

Requirement: REQ-008

Depends on: SPEC-54

## Summary

Нулевой выстрел (SPEC-54) снял ставку на готовые веса: IoU 0.22, ошибка 31 %.
Но ошибки Prithvi **некоррелированы** с нашими членами (Q 0.33–0.38 против
0.68–0.90 между своими) — единственный источник «другой информации», найденный
за две недели, и смесь принимала только такую. Решение пользователя: дообучить
кодировщик Prithvi-EO-2.0-300M (304M параметров, предобучение на всём HLS) на
наших чипах и измерить как член смеси. Это обучение с предобученным кодировщиком,
а не докрутка чужих весов; у соседа малый кодировщик с внешним предобучением
(SPEC-34) проигрывал базе — здесь кодировщик на два порядка больше и предобучен
без меток на глобальных данных, поэтому исход не предрешён.

Реализация `scripts/prithvi_finetune.py`: чекпойнт BurnScars (кодировщик +
UNet-декодер), голова заменена на 4 класса; вход — сцена «после», 6 полос
(B2 B3 B4 B8A B11 B12) / 10⁴, их means/stds; потеря как у сиама (CE с весом
фона 0.25 + dice), отражения, BF16, AdamW (кодировщик 5e-5, остальное 5e-4,
OneCycle, 60 эпох, батч 8); те же групповые фолды соседа (fit/evaluation из
его манифеста), кэш вероятностей в формате `bs-confirm-*`. Замер
`scripts/exp_prithvi_member.py`: один; DF/Q с оптикой, сиамом, бустингом; смесь —
R1 вместо оптики, R2 третьим поровну, R3 третьим с весом 0.25.

**Предзаявлено**: сигнал остановки после фолда 0 — сеть одна ниже оптики
соседа (0.618 взв.) на 0.03 или DF с оптикой выше 1.9 % (не «другая»). Принято
как член, если вариант, выбранный по фолдам 0–2, даёт пул 5 фолдов ≥ v22 + 0.004
**и** проверка 3–4 ≥ 0 **и** потерянных не больше; затем скрининг на 35 чипах
(обучение на сплите без фолда) — минимакс. Не делается сейчас: две даты как два
кадра (Prithvi умеет `num_frames`; отдельный шаг, если одна дата пройдёт),
радар и SWIR-индексы на вход (модель предобучена на шести полосах HLS),
замораживание кодировщика (полное дообучение — базовый режим их же рецепта).

## Acceptance Criteria

- Smoke на 4 чипах даёт кэш (2, 512, 512, 4); пять фолдов `research/bs-confirm-prithvi-f{f}-v1`.
- `exp_prithvi_member.py`: один, DF/Q, три варианта смеси с выбором и проверкой, потерянные.
- Принято → скрининг 35 чипов, финальная модель на 224 чипах, `inference.py` с новым членом (терраторч в зависимостях), v23.

## Verification

- На k8plus: очередь `/tmp/queue_prithvi.sh`; `.venv/bin/python scripts/exp_prithvi_member.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
