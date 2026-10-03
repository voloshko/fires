# SPEC-97: Консервативная нижняя граница покрытия событий L2A

Status: planned
Requirement: REQ-008
Depends on: SPEC-93, SPEC-96

## Summary

В523 HLS-окнах найден основной MTBS-event,17unknown. Не придумываем их IDs
и не вырезаем из исходного fit. Доказательство охвата возможно через сохранение
всех неизвестных положительных пикселей: если все дополнительные неизвестные
события U сохранены, (R+U)/(K+U)>=R/K. Порог SPEC-93 остаётся70% событий.
Это новый способ доказать тот же критерий, не изменение accuracy acceptance.

## Acceptance Criteria

- Из immutable registry SPEC-96 и официальных MTBS-полигонов строится
  pixel proof:0не положительная метка,1положительные пиксели основной
  однозначно сопоставленной гари,2защищённые неизвестные пиксели. Для unknown
  ВСЕ исходно валидные positive pixels относятся к2; для matched к2 относятся
  остаточные pixels вне основного полигона, даже если их меньше5%.
- Pixel proof привязан к hashes raster/mask/vector/registry; реестр исходных
  событий и source samples не уменьшается при failed download/QC.
- L2A-ready требует, чтобы ВСЕ protected pixels каждого исходного окна,
  включая rejected/download-failed окна, остались известными в общем valid.
  Простого сохранения файла или одного positive pixel недостаточно.
- Основной известный event считается retained только при surviving GT pixels
  внутри его официального полигона. MTBS IDs HLS и fresh нормализованы едино
  (MTBS:event_id); repeated source windows/event IDs дедуплицируются.
  EMS activation и FLOGA ID остаются явно provider-event identities.
- Если protected guarantee не выполнена, original-event bound unavailable,
  ready=false и INSUFFICIENT_EVIDENCE, без ослабления cloud masks/порогов.
  Если выполнена, lower_bound=retained_known/original_known и должен быть>=.7;
  минимум300fit twins и исходные accuracy thresholds SPEC-93 остаются.
- Тесты multi-event pixel protection, пропавшего окна/облачного остатка,
  повторов событий и неизменного denominator, а также доказуемого lower bound.
- Альтернативный source metadata audit допустим отдельно: он может дать реальные
  недостающие IDs, но не угадывает их по MGRS и не отменяет protected guarantee.

## Verification

- `python -m pytest tests/test_event_retention_bound.py -q`; `make gate`.
- Реальный pixel audit/fetch на k8plus, source-bound report; никакой ML-метрики
  не подбирает правила. Не доказывает полевую истину или independent test.
