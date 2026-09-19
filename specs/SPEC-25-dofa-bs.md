# SPEC-25: DOFA с произвольным набором каналов для BS

Status: planned

Requirement: REQ-007

Depends on: SPEC-19

## Summary

DOFA — единственный из готовых кодировщиков, принимающий **произвольный набор
полос**: патч-эмбеддинг порождается гиперсетью по длине волны каждого канала.
Это позволило бы подать все десять оптических полос Sentinel-2 обеих сцен без
отбрасывания, но наши нефизические каналы — dNBR, окна, рельеф, покров, радар —
длины волны не имеют. Спека проверяет DOFA как **оптическую ветвь**: полосы
B02…B12 «до» и «после» с их длинами волн (443…2190 нм), декодер как в SPEC-23.

Контекст, общий для SPEC-22…25. Точка отсчёта — ансамбль из девяти сетей d7w32
(5 сидов Dice + 2 Lovász + 2 граничный вес), обученных с нуля, с бустингом
(вес 0.6), правилом SCL и фильтром чужих пожаров: на 35 настроечных чипах
(модели-аналоги на 144) **IoU_burn 0.7609, mIoU_sev 0.7216, взвешенно
0.7428** (receipt SPEC-19-LIVE-011). Разброс между прогонами одной
конфигурации — 0.004 взвешенно; **всё, что меньше 0.005, — шум** и не считается
прибавкой. Лучшая одиночная сеть — 11 оптических каналов, 0.7240 взвешенно
(с бустингом). Стенд — `scripts/exp_unet.py` (обучение на 144, замер на 35),
ансамблевый замер — `scripts/exp_seeds.py`; вероятности сети на 35 чипах
кэшируются в `models/exp_<tag>.tune.npy`.

Ограничение кейса: `inference.py` работает офлайн; веса любой предобученной
модели кладутся в `models/` и грузятся с диска, лицензия указывается в receipt.
Продукты FIRMS/VNP14/MOD14 для тестовых ответов запрещены — ни один из
рассматриваемых кодировщиков их не содержит.

**Ожидание:** низкое-среднее; DOFA предобучен на разных сенсорах, но не на
гарях, и его выигрыш в литературе — на классификации, а не на границах.
**Сигнал остановки:** ниже 0.7240 − 0.005 одиночно после одного сида —
`rejected`. Спека выполняется после SPEC-22 и SPEC-23: если ImageNet-кодировщик
и Prithvi оба проигрывают сети с нуля, DOFA закрывается `deferred` без прогона
— третий проигрыш той же природы информации не добавит.

## Acceptance Criteria

- Веса DOFA (ViT-B) скачаны в `models/`, лицензия указана; загрузка через
  официальный код или TorchGeo (`torchgeo.models.dofa_base_patch16_224`).
- Один прогон на 144 чипах с полосами двух сцен и списком длин волн, замер на
  35 в общей шкале; строка в `evidence/hypotheses.md`.
- Либо запись `deferred` с ссылкой на итоги SPEC-22/23 как основание.

## Verification

- `grep -i dofa evidence/hypotheses.md` — число или основание отложить.
- `python scripts/exp_seeds.py models/exp_dofa.pt` воспроизводит число.
- Чего прогон не докажет: пользу нефизических каналов внутри DOFA — они в
  эту спеку не входят по построению.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
