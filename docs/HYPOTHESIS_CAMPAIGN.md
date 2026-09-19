# Проверка новых гипотез, 2026-09-19

Рабочая ветка `codex/fire-model-hypotheses`, база `cea06ed`. Все новые модели и
данные изолированы от `~/fires` в `k8plus:~/fires-hypotheses`. Исходный train
подключён ссылкой; test позже подключён только для финального AF-кандидата SPEC-38. Собственная .venv использует существующие
пакеты через .pth, дополнительные зависимости установлены только в неё.

## Замороженные контракты

- SPEC-32: optical / raw (18 полос + 11 индексов) / siam, два сида,
  200 эпох, одинаковые fit/tune, постобработка и бустинг.
- SPEC-33: AF hard-negative mining против случайного фона при одинаковом
  бюджете 1500/чип. 5 внешних фолдов, внутренняя calibration, 84 holdout
  не используются. Второй сид — проверка устойчивости.
- SPEC-27 + SPEC-36: AF U-Net depth5 width16, 100 эпох, вложенная calibration,
  смесь 0.3..0.7 с HGB. Без TTA. Новый стенд не меняет продуктовый inference.
- SPEC-34: до 30 событий CEMS TRAIN на первом проходе (лимит контракта 100),
  минимум 20 пригодных событий, максимум 4 патча/событие. Shared encoder
  сиамской сети предобучается на post-only 9 полосах 50 эпох. Дальше локальные
  200 эпох, контроль — siam с нуля. Внешние val/test не скачиваются.
- SPEC-35: 8 fit чипов, минимум 4 с подтверждёнными дополнительными pre;
  post/разметка не меняются. Контроль равного числа шагов — исходные пары.
- SPEC-36: скрининг на известных 35 чипах не равен независимому подтверждению.
  Положительный BS кандидат требует групповых внешних фолдов development.

Порог продвижения .005; для AF также положительная нижняя граница 95%
парного bootstrap по чипам. AF fire_event_id отсутствует: независимость
событий таким bootstrap не доказана. Грид AF фактически содержит также 1.0
из-за округления np.arange; это лишняя точка с нулевым recall, не победившая
ни на одном завершённом фолде. После просмотра результатов грид не изменяется.

## Первые результаты

SPEC-33, `research/af-hard-v2`: F1 0.906583 -> 0.927927, delta +0.021343,
95% chip-bootstrap [0.011857, 0.030498]. FP 864 -> 379, FN 599 -> 707.
Повторный подсчёт из сохранённых float32 вероятностей и исходных масок
совпадает по всем 336 чипам. Второй сид — `research/af-hard-seed2-v1`.
Это вложенная оценка с внутренним fit ~214 чипов, не прежняя OOF на fit ~269.
Числа сравнивать с парным контролем этого же прогона.

SPEC-35: 5/8 дополнительных pre прошли фиксированный QC; source match
подтверждён по реальным пикселям. Результат —
`research/temporal-pre-v1/result.json`; это доступность данных, не прирост модели.

## Запуск и состояние

```sh
ssh k8plus 'systemctl --user status fires-hyp-data-queue-v2 fires-hyp-temporal-full-v2 fires-hyp-siam-confirm --no-pager'
ssh k8plus 'tail -30 ~/fires-hypotheses/logs/af-hard-v2.log'
ssh k8plus 'tail -30 ~/fires-hypotheses/logs/gpu-queue.log'
```

CPU сервисы: `fires-hyp-af-hard-v2`, `fires-hyp-af-hard-seed2`,
`fires-hyp-bs-boost`, `fires-hyp-external-train`, `fires-hyp-temporal-pre-v2`.
Старые `fires-hyp-af-hard` и `fires-hyp-temporal-pre` завершились до замера
из-за сериализации callback argparse и отсутствующего planetary-computer.
Ошибки исправлены, не выдаются за отрицательные результаты гипотез.

GPU очередь ждёт чужие `run-queue13/14`, f0..4, rn1/2, long1/2 и свободные
22000 MiB. Существующие задачи не останавливаются.

`scripts/run_hypothesis_queue.sh`: AF-сеть -> 2 × optical/raw/siam.
`scripts/run_hypothesis_data_queue.sh`: подготовка внешних данных на CPU,
затем после основной GPU-очереди 2 × temporal, внешний pretrain и 2 × siam-transfer
при выполнении QC источников. Собственные GPU задачи исполняются последовательно.

Каждый каталог результата содержит manifest, хеши данных/кода, конфигурацию,
списки fit/calibration/evaluation, вероятности, модель и итоговые метрики.
Не переиспользовать каталог результата: JSON записывается append-only.

## Проверка

```sh
make gate
ssh k8plus 'cd ~/fires-hypotheses && OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests -q -rs'
ssh k8plus 'cd ~/fires-hypotheses && .venv/bin/python scripts/verify_af_hypothesis.py research/af-hard-v2'
```

На текущем проходе: 452 passed, 6 skipped. Пропуски — исходные LIVE проверки
FIRMS/API (нет ключей) и WorldCover (нет реального кэша). Они не доказывают
работоспособность этих внешних интеграций. Новые 10 тестов проходят без skip.
CPU smoke обоих новых нейросетевых стендов прошёл, он не является замером качества.

Ни один новый сабмит пока не собран; никакое превосходство на закрытом тесте или
над чужими моделями на другом наборе не заявляется.

## Обновление после CPU этапа и review

В ветку включён baseline v13 (`b51af08`): SPEC-22 rejected, SPEC-26 implemented.
SPEC-33 implemented, receipt SPEC-33-REPLAY-001. Второй сид: F1 0.907850 ->
0.929877, delta +0.022026, интервал [0.013157,0.030456]. Дельты положительны
на всех 5 фолдах обоих сидов. Это две реализации обучения на том же разбиении,
а не две независимые выборки. Подсчёт сохранённых вероятностей перепроверен.

Новый BS-бустинг побитно совпал с историческим tune_proba_19.npz; порядок
масок совпал; fit/tune содержат 144/35 различных event IDs без пересечений.

CEMS: 30 активаций TRAIN, 53 патча 20 м. Маленькие AOI дополнены до 512,
padding target=255 исключён из CE/Dice и статистик нормализации. Фактическая
доля настоящих пикселей: min .3263, median .9553. Padding всё равно влияет
на свёртки/BatchNorm; отрицательный результат не закроет внешние данные вообще.
CPU forward/backward на восьми настоящих внешних патчах прошёл.

Архитектуры: optical 124,439,492 параметра, raw 124,444,676, siam 141,219,524.
Это сохранение большого существующего d7-контроля, а не тест маленькой U-Net.
Сравнение siam/raw не отделяет структуру от увеличения числа параметров.

Архитектурный review: WATCH. Последующая проверка на 144 development-чипах
не является полностью независимой от выбора кандидата: эти метки ранее
участвовали в обучении моделей, оцениваемых на 35. Для независимого
подтверждения нужен нетронутый набор либо вложенный отбор всего семейства.

Очередь данных теперь `fires-hyp-data-queue-v2`; каждая GPU-стадия отдельно
ждёт 22000 MiB свободной памяти. Новые 12 тестов проходят, включая проверку
нулевого градиента на padding и целостности групп подтверждения.

## Начало GPU этапа и готовый кандидат

GPU освободился 2026-09-19 12:15 МСК. AF-сеть завершена за 313 с:
HGB .906583, сеть .885227, смесь .898848. Оптимистический pooled OOF также
отрицателен (.886197/.899438); replay 336/336 без расхождений. SPEC-27 rejected,
FAIL receipt SPEC-27-REPLAY-001. Проверенный рецепт не переносится в продукт.

SPEC-38 implemented: финальный AF hard-negative HGB на всех 420, cutoff .5,
`submissions/submission_candidate_af_hard.csv`. 447 строк, 180 AF/267 BS;
93 AF-строки изменены, BS побайтно сохранены. Validator и reload PASS.
Предсказано 3874 AF-пикселя, 48 пустых чипов — описательные числа, не качество.
Отправка не выполнялась. После выбора AF-метода SPEC-38 подключает test
отдельной ссылкой только для построения этого кандидата; экспериментальные
стенды остаются на явных train путях. Исторический AF holdout включён только
в финальное обучение, его качество не измеряется.

SPEC-37 добавлена для масштабирования пилота pre-only на все 144 fit-чипа.
Минимум 50 принят до GPU-замеров; source QC дал 87 пригодных. Ветки test,
tune и holdout не расширяются. CEMS/temporal GPU-эксперименты ещё в очереди.
Для PB >=04.00 исправляется документированный offset 1000 DN; реальная
проверка BS_tr_000122 дала RMSE .10062 -> .001435 после коррекции кодирования.
Это не отвергнутая сцено-зависимая нормализация SPEC-30.

Из основной ветки включены закрытия SPEC-25 deferred и SPEC-30/31 rejected.
В исходном HANDOFF очередь была пуста; отдельная очередь этой кампании активна.

## Siamese numerical failures and current queue

Plain FP16 Siam v1 overflowed: first bad module conv.0.0 at epoch47,
finite inputs/params, FP32 max 77007.4375 versus FP16 limit65504. Fusion BN
made the exact failed batch finite (decoder input max3034 ->195.125), but
normalized FP16 v2 later failed at epoch98. Neither has an accuracy result.
Both attempts have INSUFFICIENT_EVIDENCE receipts; artifacts are preserved.

Valid Siamese screening now uses BF16 + fusion BN, catalogs
`bs-siam-<seed>-v3`. Separate `bs-optical-bf16-<seed>-v1` controls use the same
precision. Existing optical/raw FP16 runs are retained and compared only
within that precision. External pretraining and transfer use BF16 too.
Loss/input/logit checks fail immediately; later numerical failures retain
weights and the failed batch even without per-module debug hooks.

Current main queue: finish both FP16 optical/raw seeds, then two BF16
optical/Siam pairs. Data queue waits for Siam v3, then runs 5-chip temporal
pilot and external pretrain/transfer. `fires-hyp-temporal-full-v2` waits for
both queues and runs 87-chip temporal expansion (SPEC-37), minimum50 enforced.
Completed source QC is reused on resume. No old NaN artifact is scored.

Completed first FP16 seed: optical W .733105, raw W .717137; one seed is not
used to close the two-seed contract. The AF candidate and its receipts are
unaffected by the Siamese failures.


## Synchronization with v15 and SPEC-29

Main through 3891f29 is merged into this worktree. Our SPEC-38 candidate was
adopted as v14, then superseded by v15; it must not replace the current CSV.
v15 AF F1 .9515 uses pooled OOF threshold selection on 336 chips. It is not
a nested evaluation and there is no measured constant correction to .9515.
Our original .9279/.9299 nested results retain their original protocol.
Remote running experiments keep their frozen sources; merging local main
does not silently change AF exclusion behavior in those experiments.

SPEC-29 reports 95% of 1734 FIRMS points inside annotated burn and 30.3% of
GT burn farther than 1 km from a detection. Lack of detections near false
predictions supports the false-positive interpretation but cannot prove
absence of a fire or annotation completeness. It supplies no validated
input-only filtering rule; no FIRMS test retrieval is introduced.

Both raw-band seeds are complete: mean W .739857 versus optical .743786
(delta -.003929), failing the frozen +.005 screen. Raw FAIL receipt is
SPEC-32-REPLAY-003-RAW. BF16 Siam mean .747932 versus matched optical
.738573 (+.009359), passing screening only. Its paired seed effects have
opposite signs; against FP16 optical the gain is only .004146.
Five grouped development folds are queued with new seeds 20260920–20260924,
matched BF16 controls and refitted HGB. This is development revalidation,
not independent confirmation after adaptive hypothesis selection.


## Completed temporal pilot and merged-source validation

SPEC-35 rejected: five-chip temporal pilot mean W .738294 versus .743786,
delta -.005491. The 87-chip SPEC-37 experiment remains separate and queued.
Siam v13-plus-two fixed blend scores W .759056 on the historical selection
set; it is not a validated product gain.

Merged source snapshot dcb1e32 passed 463 tests, 6 existing live checks
skipped (missing credentials/WorldCover cache). It was tested in the separate
remote directory ~/fires-hypotheses-qa-dcb1e32 using the isolated dependency
environment, without changing sources of running training jobs.
