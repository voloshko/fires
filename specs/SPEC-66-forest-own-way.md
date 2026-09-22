# SPEC-66: Лес своим путём: наш U-Net на HLS Burn Scars и перенос лес ↔ степь

Status: active

Requirement: REQ-008

Depends on: SPEC-55

## Summary

REQ-008 после снятия ставки на чужие веса (SPEC-54/64/65): проверить гипотезу
«лес против степи» своими средствами. Один и тот же наш U-Net (глубина 7, ширина
32, 6 полос одной даты — B2 B3 B4 NIR SWIR1 SWIR2, 2 класса) обучается трижды:
на **лесе** (HLS Burn Scars, 540 сцен training, 30 м), на **степи** (сцена
«после» наших чипов, групповой фолд соседа) и на **обоих**; каждая модель
меряется на лесе (их validation, 264 сцены) и на степи (чипы фолда; для лесной
модели — все 144, она их не видела). Перенос степи под лесную модель — в двух
масштабах: родные 20 м и пересчёт к 30 м. Метрика — IoU гари на валидных
пикселях; для степи ещё «потеряно пожаров».

Это первые две строки стенда трёх биомов (SPEC-55); горы (FLOGA) не загружены —
стенд закрывается как partial с двумя биомами, если этот замер состоится.

**Что мы узнаем и как решаем**: (1) IoU лесной модели на лесе против
опубликованных ~0.73–0.87 — работает ли наш рецепт на их данных вовсе;
(2) перенос лес → степь и степь → лес — насколько биомы разные для одной
архитектуры; (3) **обе → обе против каждой по отдельности** — конфликтуют ли
биомы в одном кодировщике (kill signal REQ-008: степь хуже своей модели более
чем на 0.01 при выигрыше на лесе → две модели и гейт по биому). Порогов принятия
нет: спека диагностическая, её результат — таблица 3 × 2 и запись в REQ-008.
Стоимость: 1 + 5 + 5 прогонов ≈ 2–3 ч GPU (параллельно с сидами радара).

## Acceptance Criteria

- `research/biome-{hls,steppe,both}-f{f}/summary.json` с `hls_val_iou`, `steppe_iou`, `steppe_iou_30m`, `steppe_lost`; кэши вероятностей.
- Таблица 3 × 2 в Resolution и в REQ-008.

## Verification

- На k8plus: очередь `/tmp/queue_biome.sh`; сводка `python3 -c` по summary.json.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
