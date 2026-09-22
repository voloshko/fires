# SPEC-58: Фильтр компонент бледной гари: принять или отбросить пятно, которое видит только бустинг

Status: active

Requirement: REQ-007

Depends on: SPEC-42, SPEC-47

## Summary

Бледные гари: 12 из 144 пожаров потеряны целиком и в v22. Бустинг с SWIR их
видит (0.8–1.0 на истине), но по пикселю (SPEC-42, τ и N) и по чипу целиком он
заливает ложной площадью. Ресёч №3 (блок B): не пройден **уровень компоненты** —
второй классификатор принимает или отбрасывает связное пятно, которое видит
только бустинг, по признакам пятна; при n ≈ 12–14 положительных — только
ручные признаки, ≤ 3–5, регуляризованная логистика, leave-one-fold-out; ключевой
новый признак — контраст по Red/Red-edge (dNDVI, dNDRE), потому что SWIR-индексы
не отделяют гарь от скошенного поля (M ≈ 1.0), а Red/Red-edge отделяют (M ≈ 2.7).

Реализация: `scripts/exp_component_filter.py`. Рецепт — v22-аналог на фолдах
(оптика соседа, сиам с радаром, бустинг SWIR). Кандидаты — компоненты бустинга
≥ 20 пикс. без пересечения с выходом рецепта. Признаки: log площади,
компактность, контраст dMIRBI к кольцу 5 пикс., контраст dNDVI, контраст dNDRE,
средние p гари бустинга и сиама. **Предзаявленные наборы**: A = (dMIRBI, dNDVI,
log площадь); B = A + dNDRE; C = B + p бустинга + p сиама. Метка компоненты —
доля истинной гари > 0.5. Печатаются оракул (все истинные компоненты) и «принять
все» (аналог SPEC-42) как потолок и пол.

**Критерий**: пул фолдов выше рецепта на ≥ 0.003 **и** потерянных меньше **и**
35 чипов не ниже — для одного из наборов A/B/C, причём выбор набора — по фолдам,
35 чипов только проверка. Сигнал остановки: оракул сам ниже +0.005 (ловить
нечего) или ни один набор не проходит. Не делается: обучаемые признаки
компонент (n мало), правила по одному порогу (это SPEC-42).

## Acceptance Criteria

- `exp_component_filter.py` печатает число компонент и истинных, оракул, «принять все», три набора на фолдах (LOFO) и 35 чипах, потерянные.
- Принято → фильтр в `postproc`/`inference.py` (признаки без меток), v23.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 .venv/bin/python scripts/exp_component_filter.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
