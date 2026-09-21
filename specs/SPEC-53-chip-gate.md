# SPEC-53: Почиповый гейт смеси оптика/сиам

Status: active

Requirement: REQ-007

Depends on: SPEC-50

## Summary

Последний непройденный CPU-пункт ресёча №2 (блок 2, «стекинг по чипу»), отложенный
из-за риска переобучения на 144 чипах. SPEC-50 показал систематику: доля сиама
0.6–0.7 лучше на незнакомых пожарах, 0.5 — на знакомых. Если это свойство чипа, а
не выборки, гейт по признакам чипа мог бы выбирать долю. Восемь признаков без
меток: доля чистого неба, площадь гари по оптике и по сиаму, их согласие (IoU),
уверенность бустинга на его гари, медианный dNBR предсказанной гари, средняя
уверенность оптики и сиама. Цель гейта — лучшая доля на чипе по истине (сетка
0.3…0.7); модель — HistGradientBoosting глубины 2; оценка leave-one-fold-out по
пяти групповым фолдам, затем гейт на 144 чипах → 35 скрининговых. Отдельно
печатается **оракул** (лучшая доля на каждом чипе по истине) — потолок любого
гейта, чтобы понимать, есть ли вообще что ловить.

Критерий: пул фолдов > v21 (фикс. 0.5) на 0.004 **и** 35 чипов не ниже. Сигнал
остановки — иначе, либо оракул сам ниже +0.01 (тогда гейт бессмыслен независимо
от качества). Скрипт `scripts/exp_chip_gate.py`, CPU по кэшам.

## Acceptance Criteria

- `exp_chip_gate.py` печатает распределение лучших долей, оракул, LOFO-гейт по фолдам, 35 чипов и вердикт.
- Принято → `inference.py` с гейтом (признаки считаются из входов и предсказаний, без меток), v22.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 .venv/bin/python scripts/exp_chip_gate.py`; PASS — строка «КРИТЕРИЙ SPEC-53: ПРОЙДЕН».

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
