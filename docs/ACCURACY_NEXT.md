# Кампания точности — состояние 2026-10-03

Цель: проверить обученную пару дат, российскую разметку и трудный фон,
L2A-дообучение и степень по паре дат; доказать превосходство на общей проверке.
Отрицательные эксперименты сами по себе цель не закрывают. Спеки перед кодом,
коммиты по спекам, неизменные критерии, append-only receipts.

## Размещение и ограничения

Worktree `/Users/mc/.codex/worktrees/fire-accuracy-next/fires`, ветка
`codex/fire-accuracy-next`, исходный main `1f0ea52`.
**Все вычисления и скачивания — ssh k8plus**, `/home/mc/fires-accuracy-next`.
На Mac только код и компактные доказательства. Данные в `data-store/`, веса
и результаты в `research/`. На k8plus свободно около16ГБ: новые данные ограничивать.
Старый `/home/mc/fires` и его данные читаются без изменений через symlinks.
`.runtime` — собственный venv; `.venv` ведёт в старую среду, ничего туда не ставить.
Свои задания omarchy остановлены; остаток прерванной копии сохраняется.

Один собственный GPU job, `flock research/accuracy-gpu.lock`, старт после
memory.free>=22000MiB. Не трогать соседние задания. Существующие модели и
receipts не переписывать. Native checkpoints: один tensor-файл и связанные
с ним JSON metadata, без дублирования весов.

## Завершённые измерения

**SPEC-91 rejected**, все6full models120epochs и scorer завершены.
141окно/125групп: pair IoU0.785478, post0.761802, C1-MM0.757951.
Pair−C1-MM=+0.027527, CI95[0.006894,0.055348]. Общий прирост реален,
но FLOGA lower=-0.053718 ниже-0.03 и потерянных событий2 против1 у C1-MM.
Предзаданные safeguards не пройдены. Receipt SPEC-91-REPLAY-001 FAIL,
`research/forest-pair-v1-result.json`. Не переключать продукт по pooled gain.

**SPEC-94 rejected** для inverse-sqrt весов, все6full models и scorer завершены.
172окна/172события. Balanced mIoU0.363802, hybrid0.308888:
+0.054914, CI95[0.021376,0.092000], все3класса лучше гибрида.
Equal weights0.367197; balanced−equal=-0.003395. Условие «не ниже контроля»
не выполнено. Receipt SPEC-94-REPLAY-001 FAIL,
`research/paired-severity-v1-result.json`. Оба парных варианта дают сигнал
улучшения степени; выбор equal после этого результата требует новой спеки
и свежих событий. Контур фиксирован: binaryIoU0.453463 у всех вариантов.
MTBS — спутниковая интерпретация, не наземная истина.

## Российское дообучение — SPEC-92 active

Источник PANGAEA.939429, CC-BY-4.0, визуальная Sentinel-2 разметка с GIS-проверкой;
авторы использовали FIRMS вспомогательно. Не field truth.
Исходник SHA256 `5f1cfb843d5579d800a1e8359a017f8cd33c5658f6e0c0a6eda7ac534a1c6b3c`.
196673контуров, целевой bbox80–115E/50–65N:29071. Источник содержит geometry/area,
не точные даты. Пары interval январь–15мая2020: postMay16–July15 2020,
pre тот же сезон2019, PC Sentinel-2 L2A, одна сетка/MGRS, строгий SCL.
Нельзя выдумывать точную ignition date.

Готовы447окон/127групп:309fit/61cal/77eval,24evalclusters.
Frozen reserve_plan/frame_selection, actual frames purged against prior2658
и между partitions до predictions. Geometry IDs из rawZIP, GPKG меняет ring order.
Данные `data-store/russia/manifest.json`.

Очередь `fires-accuracy-russian-transfer-v1` (MainPID742993) **обучает на GPU**.
Первый control-s1 дошёл до29/30эпох на последнем осмотре. Далее reserve-s1,
control/reserve-s2/s3, общий same-pixel scorer C1-MM+Prithvi.
Control продолжает SPEC91 pair на base374fit; reserve добавляет309Russianfit.
Одинаковый бюджет30epochs, batch8, шаги по base, AdamW3e-5/cosine/BF16.
Mining top25%FP только Russianfit, weight2, без удаления positives.
Initializer SHA/source/manifest/recipe сверяются с frozenSPEC91result.
Recipe `349b9c51f319432e707467c4ae25ce0d55db28050506e055c621346e391ca30f`.
Результат ожидается в `research/russian-transfer-v1-result.json`.

Только CI lower>0 **к обоим** C1-MM и Prithvi разрешает combined superiority.
Prithvi full512CPU preflight прошёл с официальными весами, accuracy не доказывает.
TerraTorch from_config принимает stringpaths; backbone_pretrained=False перед
strict полным checkpoint load. Targeted code review правок APPROVE;
общий независимый architecture verdict пока не закрыт.

## L2A — SPEC-93/96/97 active

Planetary Computer — основной источник. Отдельный ранний EarthSearch PREP
не смешивается с PC. PB<04 offset0; PB>=04 offset−0.1, scale0.0001;
неизвестный baseline fail-closed. Полный PC collector terminal exit2:
1008окон,889fit/119eval,861/886 известных событий сохранены.

Registry SPEC96:523/540 HLS matched MTBS,17unknown, prior2658frames.
SPEC97 защищает все GT-positive pixels неизвестных/остаточных частей;
если все сохранены, (R+U)/(K+U)>=R/K даёт честную нижнюю границу70%.
2784protected pixels потеряны в16окнах. Lower bound unavailable,
ready=false; **L2A fine-tuning не запускался**. Это нехватка data evidence,
не отрицательный результат модели. Receipt SPEC-97-BOUND001 INSUFFICIENT_EVIDENCE.

Причины в `evidence/experiments/accuracy-next-20261003/l2a-protected-pixel-causes.json`:
T11SKB1948pixels SCLcloud/shadow; T12SVC802nodata; остальные cloud/snow/missing.
PC same-day не даёт альтернативных продуктов; T10SGD20211102 отсутствует.
Публичный CDSE catalog показывает online N0500 counterparts для трёх окон,
но скачивания нет: credentials не обнаружены в проверенных конфигурациях.
Запрошены только наличие/путь конфигурации на k8plus, без секретов в чате.
Не снимать SCL, не выкидывать unknown из denominator, не снижать70%.

## Проверка и следующий шаг

SPEC90/95 implemented: isolation, provenance, grouped bootstrap, source-bound
caches, restricted legacy loading, tensor-bound metadata, exact recipe lineage.
`make gate` зелёный; это не доказательство accuracy acceptance.

```sh
ssh k8plus 'systemctl --user show fires-accuracy-russian-transfer-v1 -p ActiveState -p SubState -p MainPID -p ExecMainStatus'
ssh k8plus 'journalctl --user -u fires-accuracy-russian-transfer-v1 -n 12 --no-pager'
ssh k8plus 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

Завершить Russian6runs и обе competitor comparisons. Подготовить независимое
подтверждение equal paired severity на ранее не использованных событиях.
Ремонтировать L2A admission реальными данными/доказательствами, затем matched
fine-tune. Сохранять отрицательные outcomes, per-window counts и ограничения.
Полного доказанного превосходства и завершения всех четырёх пунктов пока нет.
