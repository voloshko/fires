# SPEC-51: Сиамская ветвь: разностная фьюжн и двухэтапная потеря

Status: active

Requirement: REQ-007

Depends on: SPEC-43, SPEC-47

## Summary

Сиамская ветвь — единственная, что откликалась на приёмы (передискретизация
+0.007, SPEC-43), и после SPEC-47…49 именно она названа резервом. Ресёч №2
(блок 6) предлагает FC-Siam-diff — разностные признаки на каждом уровне вместо
поздней фьюжн — и двухэтапное обучение «гарь/фон → степень». Наш `SiameseUNet`
уже подаёт в декодер [a, b, b−a] на каждом уровне, так что проверяются две
**вычитающие** правки, каждая отдельно:

1. `--fusion diff` — в декодер идёт только b−a (плюс 11 оптических признаков на
   первом уровне). Сеть лишается внешнего вида каждой даты и вынуждена решать по
   изменению: меньше памяти на «как выглядит эта степь», больше — на «что
   изменилось». Параметров меньше.
2. `--two-stage` — потеря из двух этапов в одной сети: бинарная CE + dice по
   логиту гарь/фон (logsumexp классов 1–3 минус класс 0) и CE степени только на
   пикселях истинной гари. Фон не тянет степень, степень не тянет фон.

Обе — сиам, BF16, `--oversample-faint 0.17 3`, сиды 20260930+f, пять групповых
фолдов **парно** к `bs-confirm-siam-f{f}-over-v1`; замер в рецепте v21 с заменой
сиама на вариант (`scripts/exp_siam_diff.py`). Ожидание: +0.003…+0.010 у одной
из правок; сигнал остановки — пул не выше парной базы на 0.004, или потерянных не
меньше, или после первого фолда −0.02 и хуже (очередь останавливается). Что не
делается: «узкая и глубокая» сеть (width 24 / depth 8) — отдельная ось, при
144 чипах шум сида ±0.02 на фолд не даст её отличить; предобученные
CD-трансформеры (веса RGB, внешнее предобучение уже проигрывало).

## Acceptance Criteria

- `SiameseUNet(fusion='diff')` даёт ту же форму выхода с меньшим числом параметров; `two_stage_loss` конечна и меньше на верном ответе (`tests/test_siam_fusion.py`).
- Бандл модели хранит `fusion` и `two_stage`; `src/comp/unet.load` строит сеть по `fusion`.
- Каталоги `research/bs-confirm-siam-f{f}-diff-v1` и `…-2st-v1`, пять фолдов каждый; `exp_siam_diff.py` — пул, фолды, чистое небо, потерянные.
- Принято → скрининг на 35 чипах (тот же флаг, сплит без `--fold`, сид 20260918), затем финальные сиамские сети и v22/v23.

## Verification

- `python -m pytest -q tests/test_siam_fusion.py`.
- На k8plus: `.venv/bin/python scripts/exp_siam_diff.py` после пяти фолдов; PASS — пул > база + 0.004 и потерянных меньше.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
