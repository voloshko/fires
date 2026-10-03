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
- SPEC-93 planned: критерии записаны, реализация L2A-дообучения ещё нужна.
- SPEC-94 active: отдельная трёхклассовая парная голова, fit-only weights,
  ignored pixels в обоих loss terms, фиксированный контур. 5 тестов.
  Реальное обучение/замер ещё нужны. При оценке MTBS0/1/5 — известный фон,
  хотя в **обучении степени** они игнорируются; не потерять этот смысл.

Всего28 новых тестов прошли на k8plus (последние5 отдельно от первых23).
make gate зелёный. Нет измеренного прироста точности этой новой кампании.

## Запущенные handles

На момент записи проверены живыми:

- `fires-accuracy-pairs-k8-v1.service`, PID610491: CPU/сетевой сборщик
  `scripts/prepare_forest_pairs.py`, data-store/forest-pairs. Продолжает
  ранее подготовленные окна по их hashes. Итоговый manifest.json создаётся
  только после всех jobs и validate_manifest. Не запускать дубликат.
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
