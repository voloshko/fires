# SPEC-22: Предобученные на ImageNet кодировщики в U-Net для BS

Status: active

Requirement: REQ-007

Depends on: SPEC-19

## Summary

Проверить рецепт победителя бенчмарка CEMS-Wildfire (U-Net с кодировщиком
MiT-B0, предобученным на ImageNet, IoU 0.779 на Sentinel-2): даёт ли
предобучение выигрыш на нашей выборке в 144 чипа, где сеть с нуля уже
на уровне CEMS. Реализация — `segmentation_models_pytorch` (`smp.Unet`,
`encoder_weights="imagenet"`, `in_channels` = число наших каналов; первая
свёртка расширяется усреднением весов). Стенд уже принимает `ARCH=smp:<enc>`.

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

**Ожидание:** среднее. Предобучение обычно помогает при малой выборке, но
ImageNet-фильтры под индексы, радар и рельеф переносятся плохо; на оптических
каналах шанс выше. **Сигнал остановки:** если ни один из четырёх кодировщиков
(MiT-B0 и MiT-B2 на оптике, MiT-B0 на 19 каналах, ResNet34 на оптике) не даёт
одиночной сети выше 0.7240 + 0.005 взвешенно **и** добавление лучшей пары к
девятке не даёт +0.005 — ось закрывается `rejected`.

## Acceptance Criteria

- Четыре прогона `ARCH=smp:mit_b0|mit_b2|resnet34` с `CHANNELS=optical|all`
  на 144 чипах, замер на 35: строки «одна» и «накоплено» из `exp_seeds.py`
  записаны в `evidence/hypotheses.md` с числами.
- Сравнение с сетью с нуля той же ширины входа — в одной таблице, в одной
  шкале (микро, правило SCL, фильтр чужих пожаров).
- Если принято: `src/comp/unet.load` грузит `arch: smp:*` без сети (веса
  ImageNet при загрузке не качаются — `encoder_weights=None`), тест на это
  есть и зелёный; финальные версии обучены на 224 и вошли в сабмит с receipt.
- Лицензия `segmentation_models_pytorch` (MIT) и весов кодировщика указана в
  receipt.

## Verification

- `grep -A3 "smp:" evidence/hypotheses.md` — четыре строки с числами.
- `python -m pytest tests/test_unet_load.py -q` — зелёный, включая загрузку
  `arch: smp:*` без сети.
- `python scripts/exp_seeds.py models/exp_smpb0opt.pt ...` на k8plus
  воспроизводит числа из журнала с точностью до 0.001 (кэш вероятностей).
- Чего прогон не докажет: перенос на закрытую выборку; сравнение честно только
  на 35 настроечных чипах.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
