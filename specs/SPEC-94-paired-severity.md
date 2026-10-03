# SPEC-94: Степень по паре дат внутри фиксированного контура

Status: planned
Requirement: REQ-008
Depends on: SPEC-90, SPEC-89

## Summary

C1-S почти всё относит к слабой степени (SPEC-86). Гибрид C1-MM+dNBR даёт
mIoU_sev0.279, но градация грубая. Учим степень по паре дат, отдельно от
контуров: фиксированный C1-MM по post, внутри — голова степени на трёх
классах, с fit-only балансировкой и неизвестными пикселями вне loss.

## Acceptance Criteria

- Данные MTBS fresh1/fresh3/fresh4 для fit (как C1-S); evaluation fresh2
  (178 пригодных окон тяжести до дополнительного QC). Нет переноса окон
  проверки в обучение, группы event_id и spatial isolation SPEC-90.
  Минимум100 пригодных evaluation окон и200 fit; иначе insufficient evidence.
- Вход:6 полос pre+6 post, одинMGRS/сетка. КлассыMTBS2/3/4=>0/1/2 головы;
  MTBS0/1/5/6 и неизвестные пиксели не обучают степень. Известный фон
  обрабатывает отдельно фиксированная модель контура C1-MM.
- Общий кодировщик пары, fusion [pre,post,diff], depth7 width32, три класса.
  120эпох AdamW3e-4 OneCyclemax1e-3, batch8 BF16, C1 augmentation. Сиды1/2/3.
  Class weights = inverse sqrt fit counts, нормировка средним и cap4,
  вычислены только по валидным размеченным fit pixels. CE плюс mean Dice
  по присутствующим в батче классам; батчи без степени не создают фоновые метки.
- Контроль: такой же paired model и расписание, но равные class weights.
  Гибрид dNBR остаётся фиксированным конкурентом. Порог контура0.5/TTA8
  и masks одинаковы. Отдельно качество степени на истинной гари и итоговый
  mIoU классов1–3 по всем валидным пикселям (как SPEC-89).
- Принятие по evaluation: delta mIoU>=.02 и CI95 lower>0 к гибриду и
  не ниже парного контроля; ни один класс не просел >.02 к гибриду.
  Bootstrap2000 по событиям seed20261003, confusion/perclass/counts сохраняются.
  Замер на прежнем fresh2 — development; проверка свежих будущих событий
  нужна до внешнего заявления превосходства. Кипр без меток — демонстрация.
- Тесты mapping/ignore, fit-only weights, оба loss terms, границы fixed mask,
  save/load, prediction geometry. Никакой успешной приёмки по smoke.

## Verification

- `python -m pytest tests/test_paired_severity.py -q`; `make gate`.
- Загрузка пар на omarchy, GPU на k8plus, systemd, 1 собственный job.
- Receipts: satellite-label agreement only. MTBS — интерпретация
  dNBR/RdNBR, не наземная истина и не проверенная степень для Сибири/Кипра.
