# SPEC-95: Безопасные веса и проверка общего рецепта экспериментов

Status: planned

Requirement: REQ-008

Depends on: SPEC-90, SPEC-91

## Summary

Независимое ревью нашло неполный список исполняемых зависимостей кэшей и
отсутствие проверки общего рецепта между последовательными прогонами.
Скореры также используют unrestricted pickle до проверки метаданных.
Добавляем tensor-only представление весов, привязанное к checksum, и проверку
общего recipe/source identity. Нынешнее обучение SPEC-91 не прерывается:
не меняем его обучающие функции или формат существующих raw-артефактов;
завершённые raw-бандлы преобразуются отдельным ограниченным загрузчиком.

## Acceptance Criteria

- `src/comp/accuracy_artifacts.py`: loading только weights_only=True;
  legacy-adapter допускает только явно перечисленные NumPy-типы и TorchVersion,
  без unrestricted pickle. Требует ожидаемый SHA256 raw-бандла до загрузки.
- Экспорт: tensor-only state + JSON metadata, source SHA и state SHA,
  schema_version. Existing bound export не перезаписывается; любое изменение
  raw/state/metadata или конфликт recipe вызывает ошибку до инференса.
- Scorers SPEC-91/93/94 и init SPEC-93 используют ограниченный путь.
  Качество и training semantics текущих SPEC-91 прогонов не меняются.
- Один список зависимостей включает scripts/train_unet.py (UNet/block),
  forest_pair.py, accuracy_contract.py и hls.py для TTA. Собственные trainer/
  scorer/variant-specific модули также включаются. Отсутствующий hash — блокер.
- Перед acceptance проверяется общая версия source_hashes/software, fit/eval
  ids, manifest и параметры рецепта всех контролей/кандидатов/сидов; допустимы
  лишь заранее заявленные различия variant, seed и fit-only normalization.
  Для новых запусков рецепт запечатывается до первого job; для уже идущего
  SPEC-91 проверяется архив каждого прогона, без подмены исходников задним числом.
- Тесты: произвольный pickle отвергнут, tamper state/raw/metadata отвергнут,
  tensor/state roundtrip, смена train_unet.py меняет подпись, source/recipe
  drift между сравниваемыми runs отвергнут, expected per-variant различия
  разрешены. Не доказывает безопасность сторонних публичных model archives.

## Verification

- `python -m pytest tests/test_accuracy_artifacts.py -q`; `make gate`.
- Ограниченный export реального завершённого post-s1 и C1-MM на k8plus;
  никаких весов или больших данных на Mac.
- Receipt SYNTHETIC/REPLAY, non_claims: не прирост точности, не исходная
  независимость разметки и не доказательство полного model-training lineage.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
