# SPEC-44: Передискретизация бледных гарей для оптической ветви

Status: active

Requirement: REQ-007

Depends on: SPEC-43

## Summary

SPEC-43 дала +0.006 парно на обеих шкалах для сиамской ветви. Оптическая ветвь
(пять сетей на 11 каналах, половина веса смеси) обучена без передискретизации
и слепа к бледным гарям так же. Цель — то же правило показа (`--oversample-faint
0.17 3`) для оптической сети, парно к оптическому сиду соседа на пяти групповых
фолдах (`research/bs-confirm-optical-f{f}-v1`, сид 20260920+f), затем в рецепте
v20 с передискретизованной сиамской ветвью. Ожидание +0.003…+0.008; сигнал
остановки — пул не выше парной базы на 0.004 или минус на 35 чипах.

## Acceptance Criteria

- Пять фолдов `research/bs-confirm-optical-f{f}-over-v1`; `scripts/exp_opt_over.py`
  — пул и фолды, парно, в рецепте v20; проверка на 35 чипах скрининговым сидом.
- Принято → пять финальных оптических сетей с передискретизацией и v21.

## Verification

- `python scripts/exp_opt_over.py` на k8plus после `/tmp/queue_opt_over.sh`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
