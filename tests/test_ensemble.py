"""Тесты ансамбля по гари (SPEC-19)."""

import numpy as np
import pytest

from src.comp.ensemble import NET_WEIGHT, predict


class ПостояннаяМодель:
    """Бустинг-заглушка: возвращает одни и те же вероятности для всех пикселей."""

    def __init__(self, proba):
        self.proba = np.asarray(proba, dtype=np.float64)

    def predict_proba(self, x):
        return np.tile(self.proba, (len(x), 1))


def test_вес_вне_диапазона_отвергается():
    with pytest.raises(ValueError, match="вес сети"):
        predict(None, None, None, net_weight=1.5)
    with pytest.raises(ValueError, match="вес сети"):
        predict(None, None, None, net_weight=-0.1)


def test_вес_по_умолчанию_в_плато():
    """0.6 выбран по замеру; плато 0.5…0.7. Тест ловит случайный сдвиг."""
    assert 0.5 <= NET_WEIGHT <= 0.7


def test_заглушка_бустинга_даёт_ожидаемое():
    m = ПостояннаяМодель([0.1, 0.2, 0.3, 0.4])
    out = m.predict_proba(np.zeros((5, 19)))
    assert out.shape == (5, 4)
    assert out[0].argmax() == 3
