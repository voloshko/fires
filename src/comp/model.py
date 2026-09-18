"""Обучение и применение модели степени поражения (SPEC-19).

Пиксельный градиентный бустинг: признаки из features.stack, цель — класс
severity 0..3. Обучение идёт только по чипам, где есть сцена post: без неё
dNBR не определён, и подавать такой чип в обучение значило бы учить модель
на нуле вместо сигнала.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rasterio.errors import RasterioIOError
from sklearn.ensemble import HistGradientBoostingClassifier

from .chips import BsChip, BsDataset
from .features import NAMES, stack
from .postproc import MIN_BLOB, drop_small

PIXELS_PER_CHIP = 18000
SEED = 20260918


def sample_chip(chip: BsChip, rng: np.random.Generator, n: int = PIXELS_PER_CHIP):
    """Пиксели чипа, сбалансированные по классам: гарь занимает малую долю."""
    features = stack(chip)
    valid = chip.valid()
    labels = chip.mask
    # Равные доли. Перевес фона проверялся трижды и проигрывал каждый раз:
    # 40/20/20/20 → 0.5159/0.4683, 55/15/15/15 → 0.5124/0.4584,
    # 70/10/10/10 (как в данных) → 0.4706/0.4228 против 0.5164/0.4733.
    # Объём тоже насыщается: 36000 → 0.4974, 72000 → 0.5036, 150000 → 0.5142.
    # Причина одна и та же — при квоте выше наличия редких классов растёт
    # только доля фона.
    quota = {cls: n // 4 for cls in (0, 1, 2, 3)}
    picked = []
    for cls in (0, 1, 2, 3):
        idx = np.flatnonzero((labels == cls).ravel() & valid.ravel())
        if idx.size:
            picked.append(rng.choice(idx, size=min(idx.size, quota[cls]), replace=False))
    if not picked:
        return np.empty((0, len(NAMES)), np.float32), np.empty(0, np.uint8)
    idx = np.concatenate(picked)
    flat = features.reshape(len(NAMES), -1)
    return flat[:, idx].T, labels.ravel()[idx]


def build_training_set(dataset: BsDataset, chip_ids: list[str], seed: int = SEED):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    unreadable: list[str] = []
    for chip_id in chip_ids:
        if not dataset.has_post(chip_id):
            continue
        try:
            chip = dataset.load(chip_id)
        except RasterioIOError:
            # Файл на диске есть, но не читается — обрыв выгрузки. Пропускаем
            # с учётом: молчаливый пропуск исказил бы состав обучающей выборки.
            unreadable.append(chip_id)
            continue
        x, y = sample_chip(chip, rng)
        if x.size:
            xs.append(x)
            ys.append(y)
    if unreadable:
        print(f"пропущено нечитаемых чипов: {len(unreadable)} ({', '.join(unreadable[:5])})")
    if not xs:
        raise RuntimeError("нет ни одного обучающего чипа со сценой post")
    return np.concatenate(xs), np.concatenate(ys)


def train(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=[NAMES.index("landcover")],
        random_state=seed,
    )
    model.fit(x, y)
    return model


def predict(model, chip: BsChip, min_blob: int = MIN_BLOB) -> np.ndarray:
    """Маска степеней поражения. `min_blob=0` отключает фильтр мелких пятен —
    он нужен при замерах, где сравнивается сырой выход модели."""
    features = stack(chip).reshape(len(NAMES), -1).T
    out = model.predict(features).astype(np.uint8).reshape(chip.shape)
    out[~chip.valid()] = 0
    return drop_small(out, min_blob)


def save(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    return path


def load(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
