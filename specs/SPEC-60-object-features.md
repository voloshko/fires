# SPEC-60: Объектные признаки пятна не из спектра: геометрия поля, текстура, радар по пятну, дата

Status: active

Requirement: REQ-007

Depends on: SPEC-58, SPEC-59

## Summary

Ресёч №4, Stage 3. Резерв «пятен только бустинга» (+0.036, 666 истинных из
5244) — отдельные пятна вдали от найденной гари (SPEC-59), спектральные
контрасты к кольцу их не разделяют (SPEC-58, AUC ≤ 0.56). Остаётся информация,
которой нет ни у одного члена: **геометрия** (заполнение bbox — прямоугольное
поле против рваной гари; доля границы пятна, совпадающей с границами полей по
Sobel NDVI «до»), **текстура** внутри пятна (разброс dNBR, средняя локальная
неоднородность 3×3 — гарь пятниста, убранное поле однородно), **радар по пятну**
(ΔVV, ΔVH, Δ(VH/VV) пятно минус кольцо, VH/VV «после»), **дата** (день года
«после», разнесение пары), плюс p сиама (AUC 0.71, SPEC-58). Ограничение: без
GLCM (заменена локальной неоднородностью) и без внешних слоёв границ полей —
границы считаются по NDVI самого чипа.

**Предзаявлено**: наборы D (не из спектра), E (D + p сиама), F (всё); модели —
логистика и HistGB глубины 2; LOFO по фолдам; порог τ по фолдам 0–2; критерий —
пул 5 фолдов ≥ +0.004 **и** проверка 3–4 ≥ 0 **и** 35 чипов ≥ 0 **и** потерянных
меньше, хотя бы для одной пары набор × модель, причём выбор пары — по фолдам.
Печатается AUC каждого признака. Сигнал остановки: ни один признак не из спектра
с AUC > 0.6 — тогда объектная ось закрывается вместе с SPEC-58/59.

## Acceptance Criteria

- `exp_object_features.py`: таблица AUC признаков, шесть строк набор × модель с вердиктом.
- Принято → фильтр компонент в `postproc`/`inference.py`, v23.

## Verification

- На k8plus: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 .venv/bin/python scripts/exp_object_features.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
