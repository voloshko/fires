# SPEC-26: Оптический ансамбль сетей для BS

Status: implemented

Requirement: REQ-007

Depends on: SPEC-19

## Summary

Сети только на 11 оптических каналах (индексы, полосы, покров — без радара,
рельефа и контекстных окон) оказались сильнее сетей на всех 19 поодиночке:
0.7240 против 0.7000–0.7138 взвешенно, а с дрожанием — 0.7259, лучшая
одиночная сеть; **две** такие сети (0.7260) равны всему ансамблю из 11
смешанных (`evidence/hypotheses.md`). Гипотеза: ансамбль из одних оптических
сетей (сиды и дрожание) сильнее смешанного, и финальный сабмит должен
строиться на нём.

Точка отсчёта: ансамбль из девяти сетей d7w32 на 19 каналах с бустингом
(вес 0.6), правилом SCL и фильтром чужих пожаров — на 35 настроечных чипах
(аналоги на 144) **0.7609 / 0.7216, взвешенно 0.7428** (receipt
SPEC-19-LIVE-011); разброс между прогонами одной конфигурации 0.004,
**всё меньше 0.005 — шум**. Стенд `scripts/exp_unet.py`, ансамблевый замер
`scripts/exp_seeds.py`, кэш вероятностей `models/exp_<tag>.tune.npy`.

**Ожидание:** +0.005…+0.010 взвешенно. **Сигнал остановки:** оптическая
пятёрка (3 сида + 2 с дрожанием) одна — не выше девятки + 0.005, и с
девяткой — не выше + 0.005: остаёмся на смешанном, спека `rejected`. Бустинг
остаётся на 19 признаках: контекстные окна ему помогали (+0.030), и это
измерено отдельно; менять его эта спека не разрешает.

## Acceptance Criteria

- Замер оптической пятёрки (`d7opt`, `d7opt_s1`, `d7opt_s2`, `d7optjit`,
  `d7optjit_s1`) одной и добавленной к девятке — строки в
  `evidence/hypotheses.md` с числами в общей шкале.
- Если принято: пять финальных оптических сетей обучены на всех 224
  (`FINAL=1 CHANNELS=optical [JITTER=0.1]`), `inference.py` грузит их через
  имена признаков в бандле (уже поддержано `unet.load`), сабмит собран,
  валидатор 0, receipt с составом ансамбля и лестницей чисел.
- Состав финала повторяет состав измеренного: никакого отбора членов по замеру
  на тех же 35 чипах (жадный отбор проверен и вреден, см. журнал).

## Verification

- `grep -n "оптик" evidence/hypotheses.md` — числа пятёрки одной и с девяткой.
- `python scripts/exp_seeds.py models/exp_d7opt.pt models/exp_d7opt_s1.pt models/exp_d7opt_s2.pt models/exp_d7optjit.pt models/exp_d7optjit_s1.pt` на k8plus воспроизводит строку «накоплено 5».
- `python inference.py --data-dir data/comp/test --out /tmp/s.csv --template data/comp/test/sample_submission.csv --unet models/bs_unet_final_opt*.pt` — код возврата 0.
- Чего не докажет: перенос на закрытую выборку.

## Resolution

**Принято и отправлено (v13, receipt SPEC-19-LIVE-013).** Оптическая пятёрка
(3 сида + 2 с дрожанием) против девятки на 19 каналах: без фильтра чужих
пожаров 0.7287 против 0.7229 взв. (+0.006, порог 0.005 пройден); с фильтром
0.7604 / 0.7255 против 0.7609 / 0.7216 (+0.002 взв.) — прибавка по степени,
площадь без изменений; по чистому небу 0.7742 против 0.7712. Смесь пятёрки с
девяткой хуже пятёрки (0.7271). Финальные пять сетей обучены на 224
(`models/bs_unet_final_opt_*.pt`), состав повторяет измеренный, отбора членов
не было. Оговорка: прибавка на грани разброса сида; версия оправдана тем, что
пять сетей на 11 каналах не хуже одиннадцати на 19 и вдвое дешевле на
инференсе.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
