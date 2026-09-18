"""Загрузка чипов соревнования (SPEC-15).

Чип — единица выдачи и единица оценки. Все растры внутри чипа приведены к
общей сетке и попиксельно совмещены; загрузчик это проверяет, а не полагает.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# Классы SCL, непригодные для расчёта индексов (как в src/masks.py).
SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)


def read_meta(root: Path, kind: str) -> pd.DataFrame:
    """`meta.csv` лежит либо внутри каталога модуля (обучение), либо один на
    весь тест уровнем выше. Во втором случае строки отбираются по `kind`."""
    own = root / "meta.csv"
    if own.exists():
        return pd.read_csv(own)
    shared = root.parent / "meta.csv"
    if not shared.exists():
        raise FileNotFoundError(f"не найден meta.csv ни в {root}, ни в {root.parent}")
    meta = pd.read_csv(shared)
    return meta[meta["kind"] == kind].reset_index(drop=True)


@dataclass(frozen=True)
class BsChip:
    chip_id: str
    pre: np.ndarray       # (10, H, W): B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL
    post: np.ndarray      # (10, H, W) либо пусто, если сцены post нет
    aux: np.ndarray       # (3, H, W): dem slope landcover
    sar: np.ndarray       # (2, H, W): VV VH
    mask: np.ndarray | None   # (H, W) severity 0..3, None в тесте

    @property
    def shape(self) -> tuple[int, int]:
        return self.pre.shape[1:]

    def valid(self) -> np.ndarray:
        """Пиксели, пригодные для индексов на ОБЕИХ сценах."""
        ok = ~np.isin(self.pre[9], SCL_INVALID)
        if self.post.size:
            ok &= ~np.isin(self.post[9], SCL_INVALID)
        return ok

    def nbr(self, stack: np.ndarray) -> np.ndarray:
        """NBR = (B8A - B12) / (B8A + B12). Полосы 20 м, как в SPEC-7."""
        nir = stack[6].astype(np.float32)
        swir = stack[8].astype(np.float32)
        total = nir + swir
        return np.divide(nir - swir, total, out=np.zeros_like(total), where=total != 0)

    def dnbr(self) -> np.ndarray:
        if not self.post.size:
            return np.zeros(self.shape, dtype=np.float32)
        return self.nbr(self.pre) - self.nbr(self.post)


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read()


class BsDataset:
    """Каталог BS-части: data/comp/train/bs или data/comp/test/bs."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = read_meta(self.root, "bs")

    def __len__(self) -> int:
        return len(self.meta)

    def chip_ids(self) -> list[str]:
        return self.meta.chip_id.tolist()

    def has_post(self, chip_id: str) -> bool:
        return (self.root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif").exists()

    def load(self, chip_id: str) -> BsChip:
        pre = _read(self.root / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif")
        post_path = self.root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif"
        post = _read(post_path) if post_path.exists() else np.empty((0, 0, 0), np.uint16)
        aux = _read(self.root / "aux" / f"{chip_id}_AUX.tif")
        sar = _read(self.root / "sentinel1_pre" / f"{chip_id}_Sentinel-1_pre.tif")
        mask_path = self.root / "masks" / f"{chip_id}_MASK.tif"
        mask = _read(mask_path)[0] if mask_path.exists() else None

        shape = pre.shape[1:]
        for name, layer in (("aux", aux), ("sar", sar)):
            if layer.shape[1:] != shape:
                raise ValueError(f"{chip_id}: слой {name} не совмещён с pre {shape}")
        if post.size and post.shape[1:] != shape:
            raise ValueError(f"{chip_id}: post не совмещён с pre {shape}")
        if mask is not None and mask.shape != shape:
            raise ValueError(f"{chip_id}: маска не совмещена с pre {shape}")
        return BsChip(chip_id, pre, post, aux, sar, mask)


def split_by_fire(meta: pd.DataFrame, holdout: float = 0.2, seed: int = 20260918) -> dict:
    """Разбиение по `fire_event_id`, а не по чипам.

    Соседние чипы одного пожара почти одинаковы: разнесение их по разные
    стороны границы завысит оценку, и на закрытой выборке она не подтвердится.
    Детерминировано по хешу, не по порядку строк, — повторный запуск с тем же
    seed даёт тот же манифест.
    """
    fires = sorted(meta.fire_event_id.dropna().unique())
    ranked = sorted(
        fires,
        key=lambda f: hashlib.sha256(f"{seed}:{f}".encode()).hexdigest(),
    )
    n_val = max(1, round(len(ranked) * holdout))
    val_fires = set(ranked[:n_val])
    return {
        "seed": seed,
        "holdout": holdout,
        "val_fires": sorted(val_fires),
        "train": meta.loc[~meta.fire_event_id.isin(val_fires), "chip_id"].tolist(),
        "val": meta.loc[meta.fire_event_id.isin(val_fires), "chip_id"].tolist(),
    }


def write_split(split: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
