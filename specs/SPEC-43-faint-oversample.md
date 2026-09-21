# SPEC-43: Передискретизация бледных гарей при обучении сиамской сети

Status: active

Requirement: REQ-007

Depends on: SPEC-32, SPEC-41, SPEC-42

## Summary

Слепота сетей к бледной гари (SPEC-40…42): 10 пожаров из 144 честных чипов
потеряны целиком, на них p_гари сетей 0.01–0.27; медианный dNBR истины +0.15
против +0.26. Три решающих правила и аугментация выцветания отклонены с числами.
Остаётся сместить само обучение: бледные гари — четверть обучающих чипов
(медианный dNBR истины < 0.17 у 25 % из 224, `research/dnbr_truth_median.json`),
но в потере они тонут — их площадь мала и контраст слаб.

Гипотеза: передискретизация. Чипы обучения с медианным dNBR истины ниже 0.17
входят в каждую эпоху трижды (список `fit` дополняется двумя копиями). Ни
входы, ни метки, ни потеря не меняются — меняется только частота показа.
Реализация — флаг `--oversample-faint 0.17 3` в `scripts/hypothesis_bs.py`;
порог считается по меткам обучающей части фолда (метки теста не трогаются).
Сиамский вариант, BF16, пять групповых фолдов, сиды 20260930+f — парно ко
второму сиду без передискретизации (`…-s2-v1`).

Честное ожидание: +0.005…+0.01 по пулу за счёт потерянных пожаров; риск — рост
ложной площади на слабом контрасте (как у выцветания). Сигнал остановки: пул
не выше парной базы на 0.004, или потерянных чипов не меньше, или после
первого фолда −0.02 и хуже (как SPEC-41 — тогда очередь останавливается).

## Acceptance Criteria

- `--oversample-faint thr k` дополняет `fit` копиями бледных чипов; без флага —
  прежнее поведение; список и порог записаны в `oversample_manifest.json`.
- Пять фолдов `research/bs-confirm-siam-f{0..4}-over-v1`.
- `scripts/exp_siam_over.py`: рецепт v19 с ветвью «сид 1 + сид 2» против
  «сид 1 + передискретизованный сид», пул, фолды, потерянные чипы.
- Принято → финальные сиды на 224 чипах и v20 с receipt'ом.

## Verification

- `python scripts/exp_siam_over.py` на k8plus после `/tmp/queue_over.sh`.
- Не доказывается: закрытая выборка; порог 0.17 и кратность 3 взяты априори.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
