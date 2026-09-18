"""Тесты загрузки сети (SPEC-19, SPEC-20)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch нужен только для сети")

from src.comp.unet import _modernise  # noqa: E402


def test_старые_имена_весов_переименовываются():
    """Сети, сохранённые до параметризации глубины, должны читаться: переобучать
    их ради одних имён — расточительство."""
    old = {"d1.0.weight": 1, "d4.1.bias": 2, "u3.weight": 3, "c1.0.weight": 4,
           "head.weight": 5, "head.bias": 6}
    new = _modernise(old)
    assert new["down.0.0.weight"] == 1
    assert new["down.3.1.bias"] == 2
    assert new["up.0.weight"] == 3
    assert new["conv.2.0.weight"] == 4
    assert new["head.weight"] == 5 and new["head.bias"] == 6
    assert not any(k.startswith(("d1.", "d4.", "u3.", "c1.")) for k in new)


def test_новые_имена_не_трогаются():
    new = {"down.0.0.weight": 1, "up.0.weight": 2, "head.bias": 3}
    assert _modernise(new) is new, "лишнего копирования быть не должно"


def test_число_ключей_сохраняется():
    old = {f"d{i}.0.weight": i for i in range(1, 5)}
    assert len(_modernise(old)) == len(old), "переименование не должно терять веса"
