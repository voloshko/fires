# SPEC-61: Сиам для малых отдельных пятен: индексы без нормализации, blob loss, вырезки вокруг пятен

Status: active

Requirement: REQ-007

Depends on: SPEC-57, SPEC-58

## Summary

Ресёч №4, Stage 2. Сиам — единственный член, который «едва видит» истинные
пятна бустинга (p медиана 0.036, у 29 % > 0.3), и единственный признак с
сигналом среди компонент (AUC 0.71). Четыре причины слабости названы, у каждой
есть лечение; проверяются **по одному**, каждое — сиам с радаром (v22-база
`…-sar-v1`), передискретизация ×3, сиды 20260930+f, пять групповых фолдов парно:

- **61a `--no-index-norm`**: десять индексов-отношений (каналы 18–27) без
  нормализации по чипу — они уже сопоставимы между сценами, а нормализация по
  статистике чипа, на которую пятно 7×7 не влияет, гасит его контраст.
  Landcover и радарные каналы нормируются как прежде.
- **61b `--blob-loss 1`**: потеря по компонентам (Kofler et al., упрощённо:
  soft dice в расширенном bbox каждой истинной компоненты, среднее по
  компонентам), вес 1:1 к текущей CE + dice. Пропуск малого пятна даёт полный
  градиент.
- **61c `--faint-crops 256`**: к каждому батчу — половина батча вырезок 256×256
  вокруг бледных компонент (медианный dNBR < 0.17, ≥ 20 пикс.) обучающих чипов;
  передискретизация на уровне пятна, а не чипа.

Замер `exp_siam_diff.py` (пул, под маской, потеряно); дополнительно — p сиама на
истинных компонентах (`exp_component_filter.py` с новым кэшем) как прямая
мишень. **Критерий** на шаг: пул ≥ база + 0.004, потерянных меньше, чистое небо
не ниже −0.002; после первого фолда −0.02 — очередь шага останавливается.
Прошедший шаг → скрининг 35 чипов → комбинирование с следующим прошедшим.
Не делается сейчас: deep supervision и upsample ×2 / мелкая ветвь (61d, только
если a–c не дали); архитектура не трогается.

## Acceptance Criteria

- `blob_loss`: пропуск малого пятна даёт потерю > 0.4 при верном большом (тест); флаги пишутся в бандл.
- Каталоги `research/bs-confirm-siam-f{f}-{nin,blob,crop}-v1`; таблица `exp_siam_diff.py`.

## Verification

- `python -m pytest -q tests/test_siam_fusion.py`; smoke `--smoke` на CPU; на k8plus `.venv/bin/python scripts/exp_siam_diff.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
