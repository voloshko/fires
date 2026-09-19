# SPEC-27: Свёрточная сеть для детекции активного горения (AF)

Status: rejected

Requirement: REQ-007

Depends on: SPEC-17, SPEC-19

## Summary

AF-компонента весит 0.35 итогового балла — столько же, сколько IoU_burn, — а
двигалась за всё время дважды: пороги af-v2 (F1 0.5994) и попиксельный бустинг
с окном 15×15 (0.8835 на отложенных, 0.9052 out-of-fold с границей 0.985,
`SPEC-19-LIVE-005`). Признаки VNP14-подобные (медиана/MAD) не помогли. Гипотеза:
свёрточная сеть на чипе 256×256 (8 каналов VIIRS + 5 вспомогательных) увидит
форму очага и кластеры горячих пикселей, которых попиксельная модель не видит,
как это произошло на BS (сеть +0.06 к бустингу по IoU_burn).

Особенности AF, определяющие постановку: доля положительных пикселей 0.035 %,
420 обучающих чипов, `fire_event_id` пуст — разбиение по чипам
(`data/comp/split_af.json`, train/val/holdout). Метрика — микро-F1 по пулу
пикселей; порог по вероятности подбирается out-of-fold, отложенные 84 чипа
трогаются один раз.

**Ожидание:** +0.01…+0.03 F1 (+0.004…+0.01 итогового балла). **Сигнал
остановки:** сеть и её смесь с бустингом не выше 0.9052 + 0.005 out-of-fold на
336 чипах train+val — остаёмся на бустинге, спека `rejected`.

## Acceptance Criteria

- Стенд `scripts/exp_af_net.py`: небольшая U-Net (глубина 4–5, ширина 16–32)
  на 256×256, потери CE с весом положительного класса + Dice, 5-фолдовая
  out-of-fold оценка на 336 чипах; результат — F1 при лучшей границе и при
  границе, выбранной на других фолдах.
- Сравнение в одной таблице: бустинг (0.9052 OOF), сеть одна, смесь
  вероятностей сети и бустинга с весом по сетке 0.3…0.7.
- Если принято: финальная сеть на всех 420 чипах, `src/comp/af.predict`
  грузит её из `models/` с запасным путём на бустинг при отсутствии torch
  (как у BS), тест на загрузку, сабмит и receipt с числами AF покомпонентно
  (precision, recall, F1) и с фиксацией числа обращений к отложенной части.
- Ограничение кейса соблюдено по построению: входы — только каналы чипа.

## Verification

- `python scripts/exp_af_net.py` на k8plus печатает таблицу OOF; те же числа в `evidence/hypotheses.md`.
- `python -m pytest tests/test_af_baseline.py -q` зелёный, включая новый тест загрузки сети.
- `python inference.py ... --template ...` — код возврата 0; число пустых AF-чипов и пикселей огня указано в receipt рядом с прежними (46 / 3949).
- Чего не докажет: F1 на закрытой выборке; разбиение по чипам не гарантирует независимости соседних чипов одного пожара.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->

## Resolution

Проверенный рецепт отвергнут: U-Net depth5 width16, 100 эпох, batch8,
13 исходных каналов, CE [1,200] + Dice, пять внешних фолдов по 336 чипам
и отдельная внутренняя calibration. Парный HGB F1 0.906583; сеть 0.885227
(delta -0.021357, 95% chip-bootstrap [-0.030959,-0.011043]); смесь 0.898848
(delta -0.007736, интервал [-0.014489,-0.001560]).
Даже оптимистические pooled OOF максимумы ниже контроля: сеть 0.886197,
смесь 0.899438, HGB 0.906546. Пересчёт сохранённых вероятностей совпал
по всем 336 чипам, без расхождений TP/FP/FN. Receipt SPEC-27-REPLAY-001 FAIL.

Финальная сеть и интеграция не выполнялись: условие принятия не выполнено.
Это не запрет всех AF-сетей, а отрицательный результат фиксированной
конфигурации. Между тем SPEC-33 дала более сильный HGB: nested F1 .9279–.9299.
