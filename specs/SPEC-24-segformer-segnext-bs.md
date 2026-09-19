# SPEC-24: Трансформеры SegFormer и SegNext для BS

Status: planned

Requirement: REQ-007

Depends on: SPEC-19

## Summary

Проверить трансформерные сегментаторы, показавшие на CEMS-Wildfire IoU 0.767
(SegNext) и близкие числа (SegFormer). На CEMS они **уступили** U-Net с
кодировщиком MiT-B0 (0.779), а ресёрч предупреждает: трансформеры с нуля на
выборках порядка сотни изображений проигрывают свёрткам. Поэтому спека
намеренно узкая: один прогон SegFormer-B1 с ImageNet-весами кодировщика
(`smp.Segformer` или HF `SegformerForSemanticSegmentation` с адаптером первой
свёртки под 11 оптических каналов), один — SegNext-T, если доступна
реализация с весами; обе — на оптике, где сети сильнее всего.

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

**Ожидание:** низкое; спека существует, чтобы закрыть вопрос замером, а не
мнением. **Сигнал остановки:** обе сети ниже 0.7240 − 0.005 одиночно —
`rejected` после одного сида каждая, второй сид не тратится.

## Acceptance Criteria

- Два прогона на 144 чипах (SegFormer-B1, SegNext-T), замер на 35 в общей
  шкале; строки «одна» и «накоплено» в `evidence/hypotheses.md`.
- Если SegNext без готовых весов недоступен в стеке k8plus — это указывается
  явно, спека закрывается `partial` по одному SegFormer.
- Стоимость инференса (секунды на чип) указана рядом с числом.

## Verification

- `grep -i "segformer\|segnext" evidence/hypotheses.md`.
- `python scripts/exp_seeds.py models/exp_segformer.pt` воспроизводит число.
- Чего прогон не докажет: потенциал трансформеров при большой выборке — только
  их поведение на 144 чипах.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
