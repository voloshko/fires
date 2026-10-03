# SPEC-96: Аудит событий HLS и географии прежнего обучения

Status: active
Requirement: REQ-008
Depends on: SPEC-90, SPEC-92, SPEC-93

## Summary

Ревью выявило: HLS-сцена имеет MGRS, но это не fire_event_id; долю удержанных
тайлов нельзя называть долей событий. Российская новая проверка также
требует явного реестра географии прежних данных. Поставляем оба source-bound
аудита до новых выводов. Неустановленный ID или география —
INSUFFICIENT_EVIDENCE, не выдуманное событие и не «нет пересечения».

## Acceptance Criteria

- `scripts/audit_training_lineage.py`: читает имеющийся публичный MTBS-vector
  mtbs_perims_DD и полный HLS Burn Scars training pool. Связь с событием
  принимается, только если официальный полигон покрывает >=95% положительных
  валидных пикселей маски, дата возгорания не позже снимка и не раньше450сут,
  следующий кандидат покрывает <=5%. Возможны unknown, никогда tile->event.
  Предзаявленные пороги геометрического аудита не подбираются по модельным метрикам.
- Для каждого HLS-окна: source raster/mask hashes, acquisition, candidate
  IDs/coverage и decision; source vector components hashes. Registry immutable.
  Наличие unknown блокирует заявление полной original-event retention SPEC-93;
  публикация отдельно mapped-scene fraction не заменяет исходный критерий.
- Реестр prior_footprints: HLS training+validation, MTBS fresh1/2/3/4, EMS,
  FLOGA, соревновательные BS train coordinates из meta.csv. Все bbox строго
  WGS84 с source hash/CRS; отсутствие требуемого источника вызывает ошибку,
  не пропуск. Указываются границы: реестр предоставленных наборов, не история
  глобального unlabeled pretraining конкурента.
- SPEC-92 chip-frame selection получает prior_footprints обязательно. До
  первых predictions проверяет непересечение evaluation/calibration с prior
  frames и fit. Старый reserve_plan сохраняется; successor overlap audit
  hash-linked к нему и реестру, не перезаписывает source receipt.
- SPEC-93 readiness вычисляется по event_id реестра вместе с MTBS/EMS/FLOGA;
  любой неизвестный HLS event оставляет ready=false. Значения tile grouping
  остаются только консервативной spatial group, без claims об event coverage.
- Тесты подтверждают spatial-only/ambiguous/no-date cases -> unknown, чёткий
  unique match -> event, lineage checksum, CRS/bbox validity и prior overlap.
  Receipt не доказывает наземную истину или все возможные источники старых моделей.

## Verification

- `python -m pytest tests/test_training_lineage.py -q`; `make gate`.
- Реальный CPU-аудит k8plus на прежних raster/metadata, без новых локальных данных.
- REPLAY receipt с mapped/unknown counts, hashes и non_claims.
