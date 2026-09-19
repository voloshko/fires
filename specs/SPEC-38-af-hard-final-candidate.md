# SPEC-38: Финальный AF hard-negative кандидат поверх BS v13

Status: implemented

Requirement: REQ-007

Depends on: SPEC-33, SPEC-20, SPEC-26

## Summary

SPEC-33 прошла замороженный критерий на двух сидах. Подготовить отдельный
офлайн CSV-кандидат, заменив только AF-ответы в сданном v13. Это не отправка
организаторам и не изменение BS или основного submission.csv.

## Acceptance Criteria

- Финальные две стадии HGB обучены на всех 420 AF train-чипах: первая со
  случайным фоном, вторая с 750 hard + 750 random. Seed 20260918, остальные
  параметры как SPEC-33. Разметка прежнего holdout используется здесь только
  для финального обучения после выбора метода; качество на ней не считается.
- Порог 0.5 заморожен до чтения test: выбран по pooled OOF первого сида,
  совпал со вторым сидом. Публикуем ранее измеренную nested F1, не train F1
  и не оптимистический pooled OOF как независимую оценку.
- Модель сохраняется стандартным `src.comp.af.save`, успешно загружается
  штатным `af.predict`, совпадение маски после reload проверяется.
- Данные test только локальные VIIRS/aux. Никаких внешних API, FIRMS/VNP14/MOD14.
- В новом CSV ровно состав template (ожидается 447 строк), полный валидатор
  RLE/размеров/непересечения классов проходит. Все не-AF строки v13 сохранены
  побайтно. Числа пустых AF-чипов и пикселей огня публикуются как описание
  предсказаний, не как качество. Хеши model/base/template/test входов в manifest.
- Отдельное имя `submission_candidate_af_hard.csv`, основной submission.csv
  не заменяется и автоматической отправки нет.

## Verification

- `python scripts/build_af_hard_candidate.py --base submissions/submission_v13-LIVE-013-optical5.csv --test data/comp/test --out research/af-hard-final-v1`
- `python -m pytest tests/test_hypothesis_lab.py -q`; `make gate`.
- Receipt SPEC-38 с командами, хешами, validator status, числом изменённых
  AF-строк и проверкой сохранности BS. Не доказывает прирост на закрытом тесте.

## Resolution

Финальный HGB обучен на 420 чипах, seed 20260918, cutoff .5; штатная
загрузка проверена. Кандидат `submissions/submission_candidate_af_hard.csv`
содержит 447 строк: 180 AF и 267 BS. Изменены 93 AF-строки, BS сохранены
побайтно. Полный validator с формами и проверкой пересечений классов PASS.
Предсказания: 48 пустых AF-чипов, 3874 пикселя огня; это не метрики качества.
SHA256 CSV c6c5ca1dd5e65665efdc2776a472a50a7e34f43bba982cf71f087df3140c6dd7.
Receipt SPEC-38-REPLAY-001. Модель и CSV на k8plus; локальная копия модели
в research/af-hard-final-v1/model.pkl (не коммитится). Отправка не выполнялась.
