# SPEC-39: Фильтр чужих пожаров по согласию оптики и сиамских сетей

Status: active

Requirement: REQ-007

Depends on: SPEC-19, SPEC-32

## Summary

Фильтр чужих пожаров (`drop_far`, SPEC-19) считает главным пятном самую большую
связную область предсказанной гари и вырезает всё дальше 125 пикселей от неё.
Ревалидация сиамских сетей на групповых фолдах (SPEC-19-LIVE-017) показала его
провал: на фолде 4 сиамская сеть ставит ложное пятно крупнее настоящего, фильтр
оставляет ложное и вырезает истину целиком — четыре чипа с пересечением 0
(`BS_tr_000129/000216/000218`, пустой `000030`). Фолд 4 — единственный, где сиам
хуже оптики (−0.023 взв. у v17).

Гипотеза: у двух разных по входу сетей ложные пятна не совпадают, а настоящий
пожар видят обе. Якорем для `drop_far` берётся не самое большое пятно смеси, а
самое большое пятно **согласия** — область, где гарь ставят и оптическая ветвь,
и сиамская. Если согласия нет ни в одном пикселе, поведение прежнее.

Честное ожидание: выигрыш сосредоточен в фолде 4 (+0.02…0.04 на нём), на
остальных фолдах — ноль, так как там якорь и так верный. Сигнал остановки: пул
по 144 чипам не выше v17 более чем на 0.004 (сидовый шум) или хоть один фолд
теряет больше 0.005.

## Acceptance Criteria

- `src/comp/postproc.py`: `drop_far(pred, far_px, anchor=None)` — при заданной
  маске `anchor` главное пятно выбирается как компонента `pred` с наибольшим
  пересечением с `anchor`; без `anchor` поведение байт-в-байт прежнее (тест).
- `src/comp/ensemble.py`: `predict(..., consensus=True)` строит `anchor` как
  «гарь по оптическим сетям И гарь по сиамским»; сети различаются по
  `variant == "siam"`. При отсутствии одного из видов сетей `anchor=None`.
- `scripts/exp_siam_confirm.py` меряет на пяти групповых фолдах соседа
  (144 чипа, пул) v17 без фильтра согласия и с ним, печатает дельту по фолдам.
- Принято → v18 (`inference.py`, receipt SPEC-19-LIVE-018) только если пул выше
  v17 более чем на 0.004 взв. и ни один фолд не теряет больше 0.005.
  Иначе — `rejected` с числами.

## Verification

- `python -m pytest -q tests/test_postproc.py tests/test_ensemble_weights.py` — зелёные,
  включая тест «`anchor=None` даёт прежний результат».
- `python scripts/exp_siam_confirm.py ~/fires-hypotheses` на k8plus: строки
  `ПУЛ … v17` и `ПУЛ … v17 + согласие`, дельты по фолдам.
- Не доказывается: результат на закрытой выборке; вариант выбран на тех же
  фолдах, на которых измерен, — это записывается в `limitations` receipt'а.
<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
