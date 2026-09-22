# SPEC-63: Правило по пикселю: гарь при согласии двух слабых сигналов — бустинга и сиама

Status: active

Requirement: REQ-007

Depends on: SPEC-42, SPEC-62

## Summary

Механика провала бледных пятен в смеси: бустинг 0.8, сиам 0.3, оптика 0.0 →
среднее 0.4·0.8 + 0.6·(0.5·0.0 + 0.5·0.3) = 0.41, argmax уходит в фон. Два
согласных слабых голоса тонут в одном молчащем. SPEC-42 переопределял фон одним
уверенным бустингом (τ 0.8–0.95, отвергнуто: ложная площадь); правило на
**согласии двух** сигналов не проверялось. Здесь: пиксель чистого неба считается
гарью, если p бустинга > τb **и** p сиама > τs, независимо от оптики; степень —
argmax смеси; фильтр чужих пожаров прежний.

**Предзаявлено**: τb ∈ {0.5, 0.6, 0.7, 0.8} × τs ∈ {0.1…0.5}; выбор по фолдам
0–2; критерий тот же, что в SPEC-62 (пул ≥ +0.004, проверка ≥ 0, 35 чипов ≥ 0,
потерянных меньше). Риск — та же ложная площадь, что у SPEC-42, но порог по
сиаму должен её отсечь (ложные пятна сиам отвергает: медиана 0.002).

## Acceptance Criteria

- Таблица τb × τs, проверка, пул, 35 чипов, вердикт — `exp_two_signals.py`.
- Принято → правило в `ensemble.predict`, v23.

## Verification

- Тот же запуск, что SPEC-62.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
