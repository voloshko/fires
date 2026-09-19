# SPEC-31: Длинное обучение оптических сетей с дрожанием

Status: active

Requirement: REQ-007

Depends on: SPEC-26

## Summary

Дрожание — регуляризатор: при нём сеть за 200 эпох могла недоучиться, а
кривая потерь у финальных сетей ещё падала. «Длинное» обучение проверялось один
раз давно, на 19 каналах и без дрожания (`bs_unet_long.pt`), и было
нейтральным. Гипотеза: на оптике с дрожанием 400 эпох дают прибавку, которой
не было без регуляризатора.

Точка отсчёта — оптическая пятёрка (SPEC-26): 0.7604 / 0.7255 с фильтром
чужих пожаров, 0.7287 взвешенно без него; лучшая одиночная оптическая сеть с
дрожанием 0.7259 взв. Разброс сида 0.004; всё меньше 0.005 — шум. Стенд
`scripts/exp_unet.py`, замер `scripts/exp_seeds.py`.

**Ожидание:** низкое-среднее, +0.002…+0.006. **Сигнал остановки:** две сети
на 400 эпохах не выше двух на 200 (0.7162 / 0.7259) на 0.005 в среднем —
`rejected`; тогда же закрывается ось «длительность обучения». Стоимость —
40 мин GPU на пару.

## Acceptance Criteria

- Два сида `EPOCHS=400 CHANNELS=optical JITTER=0.1` против пары на 200:
  строки в `evidence/hypotheses.md`.
- Если принято: финальные оптические сети переобучены на 400, сабмит, receipt.

## Verification

- `grep -n "400 эпох" evidence/hypotheses.md`.
- `python scripts/exp_seeds.py models/exp_d7optlong.pt models/exp_d7optlong_s1.pt` воспроизводит числа.
- Чего не докажет: что 400 — оптимум; только что 200 — не потолок или потолок.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
