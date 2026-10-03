# Новая кампания точности, 2026-10-03

Цель пользователя сохраняется полностью: реализовать и реально проверить
обученную пару дат, российскую разметку/трудный фон, L2A-дообучение и степень
по паре дат; добиться доказанного превосходства над конкурентом на общей
проверке. Синтетические тесты и поставка отрицательной гипотезы сами по себе
не достигают этой цели. Спеки перед кодом, отдельные коммиты по спеке.

## Размещение

- Worktree `/Users/mc/.codex/worktrees/fire-accuracy-next/fires`, ветка
  `codex/fire-accuracy-next`, исходная точка main `1f0ea52`.
- Все дальнейшие вычисления и скачивания **k8plus** (последнее уточнение
  пользователя). `/home/mc/fires-accuracy-next`, `data-store/` для данных,
  `research/` для весов и результатов. На Mac — только код и компактные receipts.
- Старые данные читаются через symlinks из `data-store/external/` на
  `/home/mc/fires/external/`. Исходные файлы не изменять. Новые пары и Россия —
  собственные каталоги в data-store. На диске k8plus около 43 ГБ свободно.
- `.runtime` — собственный venv с pytest, использует существующие ML-пакеты
  через `.pth`. `.venv` — ссылка на старый runtime, не устанавливать туда пакеты.
- Свои задачи omarchy остановлены. Подготовленные пары и российский источник
  перенесены напрямую. Остаток прерванной копии старых данных на omarchy
  сохраняется; удаление не запрошено.

## Спеки и фактически выполненное

- SPEC-90 implemented: event/spatial isolation, provenance, временная
  доступность, source-bound caches, grouped bootstrap. 13 тестов.
- SPEC-91 active: ForestPairUNet, matched trainer post/pair, подготовка пар,
  очередь 3 сидов × 2 варианта. 5 модельных + 2 data теста; CUDA smoke прошёл.
  Полное обучение и acceptance scoring **ещё не завершены**.
- SPEC-92 active: PANGAEA.939429 CC-BY-4.0 получен после HTTP503/retry.
  SHA256 `5f1cfb843d5579d800a1e8359a017f8cd33c5658f6e0c0a6eda7ac534a1c6b3c`.
  Всего196673 контуров, целевой bbox80–115E/50–65N:29071. Кластеры20км:
  fit75/calibration29/evaluation24. Поля только area и geometry: точных дат
  пожаров нет. Источник покрывает январь–15мая2020; interval-пары разрешены
  спекой отдельно от first-pass. Снимки, трудный фон и обучение ещё нужны.
  source_manifest/reserve_plan/target_contours.gpkg в data-store/russia.
- SPEC-93 active: twins, matched finetune и scorer реализованы. Источник
  исправлен на Planetary Computer согласно AGENTS.md; Earth Search PREP
  остаётся отдельной историей и в новый fit не входит. PC QC4/4 прошёл.
  Регистрация событий:523/540 HLS сопоставлены с MTBS,17unknown. Полный
  event-retention gate пока ready=false, обучение L2A не запущено.
- SPEC-94 active: отдельная трёхклассовая парная голова, fit-only weights,
  ignored pixels в обоих loss terms, фиксированный контур. 5 тестов.
  Подготовка завершена:420fit/172eval, counts[5196215,807990,302790].
  Matched trainer/scorer и sealed очередь реализованы; GPU очередь ждёт
  завершения SPEC-91/его scorer. При оценке MTBS0/1/5 — известный фон,
  хотя в **обучении степени** они игнорируются; не потерять этот смысл.

Добавлены SPEC-95 implemented (restricted loading, tensor-bound JSON,
source dependency closure, checkpoint-bound recipe audit) и SPEC-96 active
(HLS event associations + prior2658frames). Артефакты новой формы экономят
место: один tensor checkpoint с встроенным digest метаданных. Current
SPEC-91 trainer не менялся; первый post/pair имеют одинаковый recipe audit.
make gate зелёный. Нет измеренного прироста точности этой новой кампании.

## Запущенные handles

Первая подготовка пар terminal success:515sample/474group,374fit/141eval.
На последнем осмотре post1/2/3 и pair1/2 complete, pair3 обучается.
Живые/очередные handles:

- `fires-accuracy-pairs-k8-v1.service`, PID610491: CPU/сетевой сборщик
  `scripts/prepare_forest_pairs.py`, data-store/forest-pairs. Продолжает
  terminal success, manifest.json готов; не запускать заново.
- `fires-accuracy-train-k8-v1.service`, PID610520: queue_forest_pair.sh,
  ожидает окончания сборщика и ready_for_training. Затем ждёт >=22000MiB
  свободной GPU и запускает post/pair по сидам1/2/3. Одна собственная GPU
  задача, flock research/accuracy-gpu.lock. Активная очередь во время
  ожидания не означает, что GPU уже обучает модель.

Проверять actual processes, exit status и лог, а не только файл/lock:

```sh
ssh k8plus 'systemctl --user show fires-accuracy-pairs-k8-v1 fires-accuracy-train-k8-v1 -p Id -p ActiveState -p SubState -p MainPID -p ExecMainStatus'
ssh k8plus 'journalctl --user -u fires-accuracy-pairs-k8-v1 -u fires-accuracy-train-k8-v1 -n 30 --no-pager'
ssh k8plus 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```

При подготовке используются MTBSfresh1 + EMS49/FLOGA40 fit; EMS70/FLOGA87 eval.
Сборщик пишет failures и точные counts, проверяет одну сетку/MGRS, pre<start<=post,
common valid pixels и новый Fmask (включая снег/тень/cirrus). Не ослаблять
минимум200 MTBS +50 средиземноморских fit pairs после результата.

## Следующая работа

1. Проверить итог pair collection, реальные sample/group counts, отсутствие
   потери набора из-за сетки/дат, и полное обучение. Обработать только
   фактические сбои; отрицательная/неполная дата не становится «не горело».
2. Написать scorer SPEC-91: ансамбли3, matched post и исходный C1-MM на тех
   же common valid pixels, TTA8, water mask, per-window counts, потерянные
   события, event-grouped bootstrap и заранее заданные thresholds. Очередь
   сама scorer пока **не вызывает**. Нельзя принять гипотезу по train loss.
3. Подготовить российские interval-пары и фиксированную отдельную проверку;
   не превращать отсутствие контура в reviewed_negative. Mining только fit.
4. Реализовать и протестировать SPEC-93 и тренировочный/eval путь SPEC-94.
   Их вычислительные очереди используют тот же GPU flock, без конкуренции.
5. Артефакты/receipts, реальные Resolution/status, make gate, per-spec
   коммиты. Пользовательская цель не закрывается без реальной общей проверки
   улучшения и всех четырёх направлений. Нынешние PREP receipts —
   INSUFFICIENT_EVIDENCE, они не доказывают accuracy acceptance.

## Продолжение после ревью

- `fires-accuracy-pair-score-v1` ожидает все6 моделей, затем GPU scorer.
  Scorer обновлён: доверяет metadata из проверенного checkpoint, а не
  отдельно редактируемому data_manifest.json; archive сверяется с checkpoint.
- `fires-accuracy-severity-train-v1` ждёт SPEC-91+score, затем6 новых severity
  моделей и итоговый scorer. Before-start recipe seal проверяется для каждого
  job; общий flock. Source code уже синхронизирован на k8plus.
- `fires-accuracy-hls-events-v1` terminal success:523matched/17unknown;
  `fires-accuracy-prior-frames-v1` terminal success:2658frames. Registry:
  data-store/hls-events.json, prior-footprints.json. Unknown не подменяется MGRS.
- Russian prior_overlap.json:0 пересечений контуров с prior2658. Это ещё
  не actual chip-frame audit. Снимки/negative mining/перенос по России нужны.
- Архитектурное ревью подтвердило математический путь устранить unknown
  event gate без выдумывания IDs или сокращения данных: если ВСЕ неизвестные
  события гарантированно удержаны, (R+U)/(K+U)>=R/K. Для этого нужно сохранить
  ВСЕ исходно валидные положительные пиксели unknown-окон И неизвестных
  остаточных частей matched-окон. Сохранить17файлов недостаточно: облака
  могут удалить отдельный пожар. Нужна отдельная спека/реализация proof gate;
  unified MTBS IDs для HLS+MTBS и полный denominator. Порог70% не ослаблять.
- Original code review REQUEST CHANGES; architect BLOCK. Source/provider,
  missing executable hash и unbound recipe archive исправлены, но coverage и
  фактическая Russian frame isolation требуют дальнейших доказательств.
  Independent review не объявлен APPROVE. Native architect profile был
  unavailable, отдельный default-agent архитектурный lane вернул evidence.
- Не прерывать действующую GPU очередь. Для последующих изменений trainers
  соблюдать recipe seal; изменение semantics требует нового парного прогона,
  а не смешивания старого контроля с новым кандидатом.

## Итог SPEC-91 и новая подготовка (последний срез)

SPEC-91 rejected: все6full runs и scorer завершены. Pair IoU0.785478,
post0.761802, C1-MM0.757951. Global delta+0.027527 CI[0.006894,0.055348],
но FLOGA lower=-0.053718 ниже-0.03 и lost events2>1. Receipt FAIL,
порог не менялся, продукт не переключён. Это положительный общий сигнал
для новой отдельной data/recipe гипотезы, а не выполненная цель.

SPEC-94 GPU очередь работает: первый equal-s1 complete, balanced-s1
запущен. Шесть сравнительных моделей и scorer по-прежнему впереди.

SPEC-92 image collector implemented, caps fit8/cal4/eval16/source-cluster,
>=1ha и round-robin size bins. Actual frames purged against prior/cross-partition
до predictions. Geometry identity берётся из raw ZIP: GPKG меняет ring order.
Russian QC4/4 реальные interval pairs прошёл. Полный сервис
fires-accuracy-russian-prepare-v1 (PID719505) живой. Данные в data-store/russia.
Не менять selection после оценки; published perimeters не field truth.

SPEC-97 planned/code-in-progress: математическая lower bound охвата событий
с сохранением ВСЕХ неизвестных и residual-positive pixels. Pixel proofs
в data-store/event-retention/manifest.json, источник registry SPEC-96;
original denominator не сокращается. Unified MTBS IDs. Четыре теста прошли.
Полный PC collector fires-accuracy-l2a-pc-prepare-v1 запущен с --retention-proof;
может terminal code2/ready=false, если хоть один protected pixel потерян.
Это не отрицательная accuracy гипотеза и не повод ослаблять SCL.
Полный output data-store/l2a-pc-v1/manifest.json, QC successor qc-bound.json.

Продолжить: закончить российскую подготовку и написать её matched transfer/
mining/evaluation; проверить retention bound и разрешить L2A обучение только
при PASS исходного data gate. Разработать отдельную улучшенную pair recipe
после SPEC91, опираясь на новые данные/свежую проверку, не на переименование FAIL.
Цель четырёх направлений и превосходства над конкурентом остаётся active.
