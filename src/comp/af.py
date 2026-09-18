"""Детекция активного горения на AF-чипах (SPEC-17, AF-ветка SPEC-19/20).

Чип — 256×256 на сетке 375 м: VIIRS I1–I5, углы съёмки, маска валидности и
вспомогательные слои. Доля горящих пикселей в обучающей части — 0.035 %,
поэтому accuracy бессмысленна, а метрика микро-усредняется по общему пулу.

Ни одно значение здесь не взято из FIRMS, VNP14 или MOD14: пороги по яркостной
температуре — физика прибора, а не продукт.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter

# ESA WorldCover: 50 — застройка, 80 — вода. Ни там, ни там природного пожара
# быть не может, а техногенная термоаномалия — может.
LC_BUILT_UP = 50
LC_WATER = 80

# Версионированный набор порогов (SPEC-7: результат несёт версию порогов).
# Подобран на ВАЛИДАЦИОННОЙ части; отложенная трогается один раз.
THRESHOLDS = {
    "version": "af-v2",
    "window": 15,        # окно фона, пикселей (≈5.6 км при 375 м)
    "i4_min": 300.0,     # абсолютный пол по I4, K
    "i4_anom": 20.0,     # превышение над фоном окна, K — решающий порог
    "dt_min": 5.0,       # I4 − I5, K
    "dt_anom": 0.0,      # превышение ΔT над фоном окна, K
    "exclude_landcover": (LC_BUILT_UP, LC_WATER),
}

NAMES = (
    "i1", "i2", "i3", "i4", "i5", "dt",
    "i4_anom", "dt_anom", "i4_bg",
    "solar_zenith", "sensor_zenith",
    "landcover", "dem", "t2m", "rh2m", "wind",
)

PIXELS_BACKGROUND = 1500   # фоновых пикселей на чип; горящие берутся все
SEED = 20260918


@dataclass(frozen=True)
class AfChip:
    chip_id: str
    viirs: np.ndarray        # (8, H, W): I1 I2 I3 I4 I5 solar_zenith sensor_zenith valid
    aux: np.ndarray          # (5, H, W): landcover dem t2m rh2m wind_speed
    mask: np.ndarray | None  # (H, W) 0/1 active_fire, None в тесте

    @property
    def shape(self) -> tuple[int, int]:
        return self.viirs.shape[1:]

    def valid(self) -> np.ndarray:
        return self.viirs[7] > 0

    def background(self, band: np.ndarray) -> np.ndarray:
        """Фон окна. Сравнение с соседями, а не с глобальной константой:
        одинаковая аномалия должна ловиться и на тёплом, и на холодном фоне."""
        return uniform_filter(band.astype(np.float32), size=THRESHOLDS["window"], mode="nearest")


class AfDataset:
    """Каталог AF-чипов: viirs/, aux/, masks/ (masks нет в тесте)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / "meta.csv")

    def __len__(self) -> int:
        return len(self.meta)

    def chip_ids(self) -> list[str]:
        return sorted(self.meta.chip_id.astype(str))

    def _read(self, folder: str, suffix: str, chip_id: str) -> np.ndarray:
        path = self.root / folder / f"{chip_id}_{suffix}.tif"
        with rasterio.open(path) as src:
            return src.read()

    def load(self, chip_id: str) -> AfChip:
        viirs = self._read("viirs", "VIIRS_I1-I5", chip_id).astype(np.float32)
        aux = self._read("aux", "AUX", chip_id).astype(np.float32)
        if aux.shape[1:] != viirs.shape[1:]:
            raise ValueError(f"{chip_id}: aux не совмещён с viirs")
        mask_path = self.root / "masks" / f"{chip_id}_MASK.tif"
        mask = None
        if mask_path.exists():
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            if mask.shape != viirs.shape[1:]:
                raise ValueError(f"{chip_id}: маска не совмещена с viirs")
        return AfChip(chip_id=chip_id, viirs=viirs, aux=aux, mask=mask)


def split_af(meta: pd.DataFrame, seed: int = SEED,
             val: float = 0.2, holdout: float = 0.2) -> dict:
    """Деление по чипам, а не по пожарам: в AF-метаданных `fire_event_id` пуст
    целиком, группировать не по чему. Порядок строк не влияет — ранг задаёт хеш.

    `train` обучает, `val` подбирает пороги и решающую границу, `holdout`
    трогается один раз.
    """
    chips = sorted(meta.chip_id.astype(str))
    ranked = sorted(chips, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    n_val = max(1, round(len(ranked) * val))
    n_hold = max(1, round(len(ranked) * holdout))
    return {
        "seed": seed,
        "grouping": "chip",   # НЕ по пожару: fire_event_id в AF-метаданных пуст
        "holdout": sorted(ranked[:n_hold]),
        "val": sorted(ranked[n_hold:n_hold + n_val]),
        "train": sorted(ranked[n_hold + n_val:]),
    }


def features(chip: AfChip) -> np.ndarray:
    """(16, H, W) float32 — порядок совпадает с NAMES."""
    i1, i2, i3, i4, i5, solzen, senzen, _ = chip.viirs
    landcover, dem, t2m, rh2m, wind = chip.aux
    dt = i4 - i5
    i4_bg = chip.background(i4)
    return np.stack([
        i1, i2, i3, i4, i5, dt,
        i4 - i4_bg, dt - chip.background(dt), i4_bg,
        solzen, senzen,
        landcover, dem, t2m, rh2m, wind,
    ]).astype(np.float32)


def predict_threshold(chip: AfChip, thresholds: dict = THRESHOLDS) -> np.ndarray:
    """Контекстный порог (SPEC-17). Абсолютный пол отсекает холодный фон,
    аномалия над окном — тёплый, слой покрова — застройку и воду."""
    i4, i5 = chip.viirs[3], chip.viirs[4]
    hot = (i4 > thresholds["i4_min"])
    hot &= (i4 - chip.background(i4)) > thresholds["i4_anom"]
    hot &= (i4 - i5) > thresholds["dt_min"]
    hot &= ((i4 - i5) - chip.background(i4 - i5)) > thresholds["dt_anom"]
    hot &= ~np.isin(chip.aux[0], thresholds["exclude_landcover"])
    return (hot & chip.valid()).astype(np.uint8)


def f1(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Микро-усреднение: пул пикселей, а не среднее по чипам."""
    t = np.asarray(truth, dtype=bool)
    p = np.asarray(pred, dtype=bool)
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": score, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def score_chips(dataset: AfDataset, chip_ids: list[str], predict_fn) -> dict:
    """Пул по всем чипам — микро-F1, как требует доля горящих пикселей 0.035 %."""
    truth, pred = [], []
    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        truth.append(chip.mask.reshape(-1))
        pred.append(predict_fn(chip).reshape(-1))
    return f1(np.concatenate(truth), np.concatenate(pred))


def build_training_set(dataset: AfDataset, chip_ids: list[str], seed: int = SEED):
    """Все горящие пиксели + случайная выборка фона. Брать фон целиком нельзя:
    при доле 0.035 % это 65 тысяч отрицательных на два десятка положительных."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for chip_id in chip_ids:
        chip = dataset.load(chip_id)
        feats = features(chip).reshape(len(NAMES), -1).T
        labels = (chip.mask.reshape(-1) > 0)
        ok = chip.valid().reshape(-1)
        fire = np.flatnonzero(labels & ok)
        back = np.flatnonzero(~labels & ok)
        if back.size > PIXELS_BACKGROUND:
            back = rng.choice(back, PIXELS_BACKGROUND, replace=False)
        take = np.concatenate([fire, back])
        xs.append(feats[take])
        ys.append(labels[take].astype(np.int8))
    return np.concatenate(xs), np.concatenate(ys)


def train(x: np.ndarray, y: np.ndarray, seed: int = SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        categorical_features=[NAMES.index("landcover")],
        random_state=seed,
    ).fit(x, y)


def predict_model(model, chip: AfChip, cutoff: float = 0.5) -> np.ndarray:
    feats = features(chip).reshape(len(NAMES), -1).T
    proba = model.predict_proba(feats)[:, 1].reshape(chip.shape)
    hot = proba >= cutoff
    hot &= ~np.isin(chip.aux[0], THRESHOLDS["exclude_landcover"])
    return (hot & chip.valid()).astype(np.uint8)


def save(model, path: str | Path, cutoff: float = 0.5) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"model": model, "cutoff": cutoff, "names": NAMES}, fh)


def load(path: str | Path):
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def predict(chip: AfChip, model_path: str | Path = "models/af_hgb.pkl") -> np.ndarray:
    """Точка входа для inference.py: модель с диска, иначе пороговый baseline."""
    path = Path(model_path)
    if not path.exists():
        return predict_threshold(chip)
    bundle = load(path)
    return predict_model(bundle["model"], chip, bundle["cutoff"])
