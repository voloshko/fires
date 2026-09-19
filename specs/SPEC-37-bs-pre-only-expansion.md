# SPEC-37: Масштабированная pre-only аугментация BS

Status: active

Requirement: REQ-006

Depends on: SPEC-35, SPEC-19

## Summary

Пилот SPEC-35 подтвердил 5/8 пригодных дополнительных pre. Пять изменяемых
чипов из 144 дают слишком слабое вмешательство для проверки сильной модельной
прибавки. До просмотра GPU-результатов расширить тот же frozen source protocol
на все 144 fit чипа, без изменения post, разметки и порогов качества.
Никакие tune/holdout/test не расширяются. Самостоятельный связный эксперимент,
а не изменение критерия уже запущенного пилота SPEC-35.

## Acceptance Criteria

- Каталог запрашивается для всех 144 исходных fit-чипов; та же проверка
  совпадения исходной pre, MGRS, SCL и NBR, что SPEC-35. До трёх ближайших
  кандидатов в исходном 30-дневном окне. Пороги не ослабляются по результату.
- Для модельного сравнения минимум 50 чипов проходят source QC. Иначе deferred
  с INSUFFICIENT_EVIDENCE, а не утверждение, что многодатность бесполезна.
- Два сида optical, 200 эпох, depth7 width32 batch8, те же шаги и геометрическая
  аугментация, что optical-контроль SPEC-32. Для доступного чипа pre заменяется
  с вероятностью .5; статистики стандартизации берутся по исходным fit.
- Рост среднего W >=.005 на 35 selection-чипах является только положительным
  скринингом; дальше development-перепроверка SPEC-36 с явными ограничениями
  адаптивного выбора. Меньшая/отрицательная дельта — rejected с FAIL receipt.
- Фиксированная смесь v13+два новых сида публикуется без подбора веса/состава.
- Source manifests, item IDs, маскированные доли, NBR QC, список пропусков,
  хеши дополнительных pre и обеих сетей остаются в артефактах.

## Source encoding correction

До GPU-замеров обнаружено и подтверждено на исходной сцене BS_tr_000122:
Planetary Computer PB 05.10/05.09 содержит offset +1000 DN, обучающие чипы
harmonized. RMSE B12 raw .10062, после документированной поправки .001435.
Для PB >=04.00 вычитаются 1000 DN из спектральных полос с отсечением у нуля;
SCL не меняется. Это правило формата ESA, не регрессия post к pre SPEC-30.
Источники: https://sentiwiki.copernicus.eu/web/s2-processing и
https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html.
Неизвестный baseline блокирует использование сцены. Пороги QC не меняются.
Первый full-pre-v1 остановлен до завершения из-за этого несоответствия
кодирования; новый источник full-pre-v2 использует сохранённый каталог и
отдельные результаты. Пилот SPEC-35 на сценах PB <04.00 не затронут.

## Verification

- `python scripts/hypothesis_sources.py temporal --limit 144 --out research/temporal-full-catalog-v1`
- `python scripts/hypothesis_temporal.py --catalog research/temporal-full-catalog-v1 --out research/temporal-full-pre-v2`
- `python scripts/hypothesis_bs.py train --variant optical --seed 20260918 --extra research/temporal-full-pre-v2 --min-extra 50 --out research/bs-temporal-full-20260918-v1` (повтор с 20260919).
- `python -m pytest tests/test_hypothesis_lab.py -q`; `make gate`.
- Не доказывает пер-пиксельную неизменность severity при другой pre, физическую
  площадь, перенос на закрытый тест или преимущество на чужом бенчмарке.
