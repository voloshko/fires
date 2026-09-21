# SPEC-41: Аугментация выцветания гари для сиамской сети

Status: active

Requirement: REQ-007

Depends on: SPEC-32, SPEC-40

## Summary

Разбор ошибок v19 на 144 честных чипах: 16 пожаров потеряны целиком, у 10 из них
сети дают вероятность гари на истине 0.01–0.27 при бустинге 0.8–1.0; медианный
dNBR их истины +0.15 против +0.26 у остальных. Сети выучили «яркую» гарь и не
видят бледную; потерянные пожары — 15 % всех ошибок. Порог (SPEC-40) это не
лечит — знак зависит от шкалы. Лечить надо обучением.

Гипотеза: аугментация «выцветание». С вероятностью ½ сети показывают копию
чипа, где спектральные каналы сцены «после» сдвинуты к сцене «до»:
`post' = pre + α·(post − pre)`, α ~ U(0.3, 0.8), SCL и метка прежние. Модель
вынуждена находить гарь по слабому контрасту. Реализация — через существующий
механизм подмены входа (`extra_map`) в `scripts/hypothesis_bs.py`, флаг
`--fade lo hi`; сиамский вариант, BF16, пять групповых фолдов соседа.

Честное ожидание: +0.005…+0.015 взв. по пулу за счёт потерянных пожаров;
риск — размытие кромки и рост ложной площади на «нормальных» гарях. Сигнал
остановки: пул не выше базы (тот же фолд, тот же сид, без `--fade`) на 0.004,
или потерянных чипов не стало меньше.

## Acceptance Criteria

- `scripts/hypothesis_bs.py --fade 0.3 0.8` строит по одной выцветшей копии на
  обучающий чип и подменяет вход с вероятностью ½; без флага поведение прежнее.
- Пять фолдов `research/bs-confirm-siam-f{0..4}-fade-v1`, сид 20260930+f — тот же,
  что у второго сида без выцветания (`…-s2-v1`): сравнение парное.
- `scripts/exp_siam_fade.py`: рецепт v19 с сиамской ветвью «сид 1 + сид 2» против
  «сид 1 + выцветший сид», пул по 144 чипам и по фолдам, число потерянных чипов.
- Принято → финальные сиды с выцветанием на 224 чипах и v20.

## Verification

- `python scripts/exp_siam_fade.py` на k8plus после очереди `/tmp/queue_fade.sh`.
- Не доказывается: закрытая выборка; α подобрана априори, не перебиралась.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
