"""RLE для submission.csv (SPEC-16).

Нумерация пикселей — построчно слева направо и сверху вниз, начиная с ЕДИНИЦЫ.
Пары идут по возрастанию старта, серии не пересекаются и НЕ СОПРИКАСАЮТСЯ
(смежные сливаются — это часть формата, а не оптимизация).
"""

from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> str:
    """Булева маска -> строка RLE. Пустая маска даёт пустую строку."""
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    if not flat.any():
        return ""
    # Границы серий: сравниваем со сдвинутым массивом, обрамляя нулями.
    padded = np.concatenate(([False], flat, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    starts = change[0::2] + 1  # +1: нумерация с единицы
    ends = change[1::2] + 1
    lengths = ends - starts
    return " ".join(f"{s} {l}" for s, l in zip(starts, lengths))


def decode(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Строка RLE -> булева маска заданной формы."""
    height, width = shape
    flat = np.zeros(height * width, dtype=bool)
    tokens = rle.split()
    if not tokens:
        return flat.reshape(shape)
    if len(tokens) % 2:
        raise ValueError("RLE: нечётное число чисел — пары неполны")
    numbers = np.asarray(tokens, dtype=np.int64)
    starts, lengths = numbers[0::2], numbers[1::2]
    if (starts < 1).any():
        raise ValueError("RLE: нумерация начинается с единицы")
    if (lengths < 1).any():
        raise ValueError("RLE: длина серии должна быть положительной")
    if (starts + lengths - 1 > flat.size).any():
        raise ValueError("RLE: серия выходит за пределы чипа")
    for start, length in zip(starts, lengths):
        flat[start - 1 : start - 1 + length] = True
    return flat.reshape(shape)
