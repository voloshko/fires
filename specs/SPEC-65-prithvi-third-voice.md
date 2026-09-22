# SPEC-65: Prithvi как третий голос смеси: критерий по смеси, проверка вслепую на фолдах 2–4

Status: active

Requirement: REQ-007

Depends on: SPEC-64

## Summary

SPEC-64 остановлена по правилу «сеть одна ниже 0.588»: дообученный Prithvi один
даёт 0.388. Но его ошибки почти не пересекаются с нашими (double-fault с оптикой
1.27 %, ниже 1.19 % между оптикой и сиамом), и третьим голосом с весом 0.25 он
дал +0.0094 на фолде 0. Это другая гипотеза — не «член силён», а «член
дополняет», — и она требует другого критерия. Он записывается здесь **до того,
как увидены фолды 2–4**.

Обучение — то же (`prithvi_finetune.py`, тот же сид-ряд, те же каталоги
`bs-confirm-prithvi-f{f}-v1`); фолды 0–1 обучены под SPEC-64 и уже увидены,
фолды 2–4 — нет. **Протокол**: вариант смеси (R2 третьим поровну / R3 третьим с
весом 0.25; R1 «вместо оптики» исключён по фолду 0: −0.024) выбирается по
фолдам **0–1**; проверка **вслепую на фолдах 2–4**: среднее Δ к v22 ≥ 0 и ни
один из трёх фолдов ниже −0.01; пул пяти фолдов ≥ v22 + 0.004; потерянных не
больше 12. Затем скрининг на 35 чипах (обучение на сплите без фолда) — минимакс,
как всегда. Честная оговорка: выбор между двумя вариантами на двух фолдах —
слабая селекция, поэтому решающая — слепая проверка.

Что не делается: замораживание кодировщика, две даты кадрами, вес > 0.25 —
любое из этого после результата было бы подгонкой; если гипотеза пройдёт, они
станут отдельными спеками. Стоимость: три фолда ≈ 30 мин GPU.

## Acceptance Criteria

- Пять каталогов `bs-confirm-prithvi-f{f}-v1`; `exp_prithvi_member.py --select 0,1 --check 2,3,4` печатает выбор, слепую проверку, пул, потерянные, вердикт.
- Принято → скрининг 35 чипов, финальная модель на 224 чипах, `inference.py` с третьим членом, v23.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/exp_prithvi_member.py --select 0,1 --check 2,3,4`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
