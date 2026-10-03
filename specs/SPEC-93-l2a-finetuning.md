# SPEC-93: Дообучение C1-MM на Sentinel-2 L2A

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-83

## Summary

SPEC-83: перенос C1-MM с HLS на L2A не прошёл допуск на Греции (CI lower -0.037
при минимуме -0.03). Дообучаем на реальном L2A той же даты/MGRS/сетки и с теми
же метками; гипотеза — устранить сдвиг источника без нормализации post к pre.

## Acceptance Criteria

- L2A берётся из Planetary Computer (`sentinel-2-l2a`), анонимно, согласно
  AGENTS.md. Ранний Earth Search QC сохраняется как отдельный PREP-артефакт
  и не входит в новое обучение/проверку. Новый каталог l2a-pc-v1.
- L2A twins для fit C1-MM: HLS540 + MTBS300 + EMS49/FLOGA40, взвешивание
  средиземноморских ×2. Наборы evaluation не добавляются в fit. Минимум
  300 fit twins, >=70% исходного fit по событиям, иначе insufficient evidence.
  События HLS определяются registry SPEC-96; MGRS не заменяет event_id,
  любое неизвестное событие блокирует заявленную original-event coverage.
- Метаданные STAC определяют bands/scales/offsets, B8A+B12 подтверждаются
  по assets. Если PC не отдаёт raster:bands, scale/offset берутся по
  официальной кодировке ESA и s2:processing_baseline (до04:0, с04:-1000 DN),
  с явным encoding_basis. Неизвестный baseline блокирует окно. SCL invalid 0/1/3/8/9/10/11; вода SCL6 и NDWI>0; сетка30м.
  Сохранение hashed inputs/labels/ids/dates; кэш tied to input+model+code.
- Сиды1/2/3 из C1-MM соответствующих сидов: полное дообучение 30 эпох,
  AdamW lr3e-5, cosine schedule, CE[.25,1]+Dice, batch8, BF16, C1 augmentation.
  Контроль: те же веса/шаги/сиды, продолжение на HLS тех же пригодных окон.
  Статистики нормализации только fit и валидные пиксели.
- Eval EMS/FLOGA L2A twins на общих валидных пикселях всех сравниваемых
  источников, TTA8, порог0.5. После обучения ансамбль3 против исходного
  C1-MM и парного HLS-продолжения. Принятие: pooled delta>=.01 и CI95 lower>0
  против обоих; каждый набор delta>=-.01 и lower>=-.03. Восстановление
  одного допуска без полезной прибавки не принимается как прирост точности.
- Тесты offsets/nodata, B8A, SCL, сетки и сохранения модели; точность smoke
  не заявляется. Контракт фиксируется до первых training outcomes.

## Verification

- `python -m pytest tests/test_l2a_finetuning.py -q`; `make gate`.
- Данные, подготовленные окна и GPU jobs на k8plus в data-store/, отдельные
  каталоги и systemd, 1 GPU job. Receipts с CI/group counts и hashes.
- non_claims: не независимый тест после адаптивного ресёрча, не мировой
  рейтинг и не подтверждение на Сибири (отдельная проверка SPEC-92).
