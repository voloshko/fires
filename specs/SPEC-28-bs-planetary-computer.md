# SPEC-28: Многодатная аугментация обучающих чипов BS из Planetary Computer

Status: planned

Requirement: REQ-006

Depends on: SPEC-7, SPEC-19

## Summary

Спектральное дрожание — имитация «того же места в другой день» — дало самую
крупную прибавку среди приёмов обучения (+0.008 на одиночной сети, 0.7259 с
оптикой). Настоящие дополнительные сцены Sentinel-2 для **обучающих** чипов
под ту же разметку — честная версия того же приёма: другие дата, освещение,
облачность, фенология при неизменной истине. Обучающие чипы несут привязку
(EPSG, x_min…y_max в `data/comp/train/bs/meta.csv`) и даты `date_pre`,
`date_post`; тестовые чипы привязки не имеют, и для теста ничего не качается
— это правило кейса и `non_claim` каждого receipt.

Источник — Planetary Computer, анонимно, как в SPEC-7 (`src/burn.py`): для
каждого чипа 1–3 дополнительные сцены «до» в окне до 30 дней перед
`date_pre` и 1–3 «после» в окне до 30 дней после `date_post`, тот же MGRS-тайл,
доля облаков по SCL ≤ 30 %. Пары «до/после» комбинируются; маска одна.
Разрешение ресёрча (SPEC-19): самостоятельный сбор данных из открытых
источников ДЗЗ для обучения разрешён.

**Ожидание:** +0.005…+0.015 взвешенно; риск — фенологический сдвиг между
сценами размывает сигнал «до/после» и добавляет ложную гарь на убранных полях.
**Сигнал остановки:** сеть на расширенной выборке не выше сети с дрожанием
(0.7259) + 0.005 при двух сидах — `rejected`; сама выгрузка остаётся как актив.
Стоимость: выгрузка ~224 × 4 сцены × 10 полос при 20 м — часы; спека
выполняется после SPEC-26/27.

## Acceptance Criteria

- `scripts/fetch_extra_scenes.py`: по `meta.csv` обучающих чипов качает
  дополнительные сцены в `data/comp/train/bs_extra/<chip>/<date>_Sentinel-2.tif`
  в сетке чипа (тот же EPSG, 512×512, 20 м, 10 полос + SCL), с журналом
  пропусков (нет сцены в окне, облачность выше порога) — число пропусков в
  отчёте.
- Стенд `exp_unet.py` с `EXTRA=1`: обучающие примеры — все комбинации
  «до/после» чипа с той же маской; настроечные 35 чипов дополнительными
  сценами **не** расширяются — они меряются как есть.
- Два сида на 144 (+ extra) против двух сидов с дрожанием: числа в журнале.
- Ни одна строка кода инференса не меняется: сеть та же, вход тот же.

## Verification

- `ls data/comp/train/bs_extra | wc -l` и журнал пропусков — сколько чипов удалось расширить.
- `EXTRA=1 CHANNELS=optical DEPTH=7 WIDTH=32 TAG=d7extra python scripts/exp_unet.py` и строка в `evidence/hypotheses.md`.
- `git diff --stat inference.py src/comp/unet.py src/comp/ensemble.py` — пусто.
- Чего не докажет: что прибавка не от простого удвоения числа примеров — контроль: тот же стенд с дублированием исходных пар без новых сцен.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
