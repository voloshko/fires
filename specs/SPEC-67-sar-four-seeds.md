# SPEC-67: Устойчивость v22: четыре сида радарного сиама

Status: active

Requirement: REQ-007

Depends on: SPEC-57

## Summary

v22 держится на двух сидах радарного сиама (55/56); на 35 чипах два сида дали
−0.0064 к v21 при разбросе сидов базы 0.7607–0.7685. Разброс сида на фолдах
±0.02. Спека измеряет, что даёт **четыре сида вместо двух**: на фолдах — ещё
два сида (20260940+f, 20260950+f) парно к имеющемуся (20260930+f): один сид,
средние двух и трёх; на 35 чипах — ещё два скрининга (20260920, 20260921):
среднее четырёх против среднего двух over-сидов; финальные сети 57/58 для
продукта. **Критерий v23**: среднее трёх сидов на фолдах ≥ v22 (один сид) + 0.002
**и** четыре сида на 35 чипах не ниже двух over-сидов (0.7723) − 0.002 — то есть
хотя бы снятие минуса на случайных чипах. Иначе v22 остаётся, спека — диагноз
разброса. ≈ 5.5 ч GPU параллельно с SPEC-66.

## Acceptance Criteria

- Каталоги `bs-confirm-siam-f{f}-sar{40,50}-v1`, `bs-siam-sar-screen-2026092{0,1}`, модели `bs_unet_final_siam_sar_s5{7,8}.pt`.
- Замер `exp_sar_seeds.py`: фолды (1/2/3 сида), 35 чипов (2/4 сида), вердикт.

## Verification

- На k8plus после очереди: `.venv/bin/python scripts/exp_sar_seeds.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
