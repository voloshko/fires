# SPEC-52: Мягкие метки у кромки гари

Status: active

Requirement: REQ-007

Depends on: SPEC-43

## Summary

Последний непройденный GPU-пункт ресёча №2 (блок 4). Дилатация и порог у кромки
отклонены (`exp_edge_folds.py`): разметчик не обводит шире предсказания. Остаётся
обратная сторона того же блока — не двигать кромку, а **не наказывать сеть за
пиксель ширины на границе**: one-hot метки усредняются окном 3×3
(`soft_edge_targets`), вдали от кромки метка прежняя, в кольце ±1 пиксель масса
делится между соседними классами. CE по мягким меткам с прежними весами классов,
dice прежний. Флаг `--soft-edge 3`, сиам с передискретизацией, пять групповых
фолдов парно к `…-over-v1`, замер в `exp_siam_diff.py` (ветка `soft`).

Ожидание низкое (+0.000…+0.004): кромка — малая доля площади, а провал дилатации
говорит, что граница у разметчика резкая. Спека закрывает пробел брифа, а не
несёт надежду. Сигнал остановки: пул не выше базы на 0.004 или потерянных не
меньше; после первого фолда −0.02 — стоп очереди. Скрининг на 35 чипах — только
при проходе на фолдах.

## Acceptance Criteria

- `soft_edge_targets`: сумма по классам 1, one-hot вдали от кромки, доли у кромки; `soft_edge_loss` меньше на верном ответе (`tests/test_siam_fusion.py`).
- Пять фолдов `research/bs-confirm-siam-f{f}-soft-v1`; таблица `exp_siam_diff.py`.
- Принято → скрининг 35 чипов, финальные сети, v22. Отклонено → Resolution, леджер.

## Verification

- `python -m pytest -q tests/test_siam_fusion.py`; на k8plus `.venv/bin/python scripts/exp_siam_diff.py`.

<!--
## Resolution (added when work lands — do not fill in advance)

Appended by the orchestrating session when the spec reaches a terminal-ish
status. Records honestly: what was planned vs what was found, deviations and
why, measured numbers, what was deliberately NOT done, and follow-up specs
opened. The Status line above is flipped ONLY together with writing this
section, and only by the orchestrating session — never by an implementing
subagent. See CLAUDE.md "Resolution convention"; SPEC-545 is a good model.
-->
