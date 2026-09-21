# SPEC-48: Карта бустинга как входной канал оптической сети

Status: active

Requirement: REQ-007

Depends on: SPEC-47

## Summary

Бустинг с SWIR-признаками (SPEC-47) видит бледные гари (0.8–1.0 на истине), сети
не видят (0.01–0.27), а смешивание на выходе с весом 0.4 их не спасает —
потерянных пожаров 14 из 144 и в v21. Стекинг вероятностей на выходе отвергнут
(SPEC-19, 35 чипов). Ресёч №2 (блок 3c) различает его и **auto-context**: карта
вероятностей бустинга подаётся сети на вход, и сеть учится пространственно
поправлять её, а не перевзвешивать.

Гипотеза: оптическая сеть с 15 каналами — 11 базовых + 4 вероятности класса
от бустинга. Без утечки: для обучающих чипов фолда вероятности считаются
out-of-fold внутри обучающей части (3 внутренних групповых фолда), для
оценочных — бустингом на всей обучающей части (уже есть,
`bs-confirm-boost-f{f}-swir-v1`). Реализация — каталог карт по чипам и хук
`EXTRA_CHANNELS_DIR` в `bs_inputs`; пять групповых фолдов, парно к оптике соседа.

Ожидание: +0.005…+0.015 и заметно меньше потерянных пожаров. Риск: сеть начнёт
копировать бустинг и наследует его ложную площадь. Сигнал остановки: пул не
выше v21 на 0.004, или потерянных не меньше, или минус на 35 чипах.

## Acceptance Criteria

- `scripts/boost_oof.py --fold f` пишет `research/boost_oof-f{f}/{chip}.npy` (H, W, 4)
  для обучающих чипов (OOF) и оценочных (fit-модель); хеши в манифесте.
- `EXTRA_CHANNELS_DIR` в `bs_inputs`: без переменной поведение прежнее (тест).
- Пять фолдов `research/bs-confirm-optical-f{f}-bch-v1`; `scripts/exp_boost_channel.py`
  — пул, фолды, потерянные, чистое небо; затем скрининг на 35 чипах.
- Принято → финальные оптические сети с каналом и `inference.py`, считающий
  бустинг до сетей; v22.

## Verification

- `python -m pytest -q tests`; `python scripts/exp_boost_channel.py` на k8plus.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
