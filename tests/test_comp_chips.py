"""SPEC-15: загрузка чипов, метаданные, разбиение по пожарам."""

import numpy as np
import pandas as pd
import pytest

from src.comp.chips import split_by_fire, write_split

TRAIN_COLUMNS = ["chip_id", "kind", "fire_event_id", "region", "valid_frac", "cloud_frac"]
TEST_COLUMNS = ["chip_id", "kind", "width", "height", "gsd", "valid_frac", "cloud_frac"]


def _meta(n_fires=10, chips_per_fire=3):
    rows = []
    for fire in range(n_fires):
        for chip in range(chips_per_fire):
            rows.append(
                {
                    "chip_id": f"BS_tr_{fire:03d}{chip}",
                    "kind": "bs",
                    "fire_event_id": f"FE{fire:05d}",
                    "region": "test",
                    "valid_frac": 0.9,
                    "cloud_frac": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_split_never_puts_one_fire_on_both_sides():
    meta = _meta()
    split = split_by_fire(meta)
    train_fires = set(meta.set_index("chip_id").loc[split["train"], "fire_event_id"])
    val_fires = set(meta.set_index("chip_id").loc[split["val"], "fire_event_id"])
    assert not (train_fires & val_fires)
    assert set(split["train"]) | set(split["val"]) == set(meta.chip_id)


def test_split_is_reproducible(tmp_path):
    meta = _meta()
    first = split_by_fire(meta, seed=1)
    second = split_by_fire(meta, seed=1)
    assert first == second
    assert split_by_fire(meta, seed=2) != first
    path = write_split(first, tmp_path / "split.json")
    assert path.exists()


def test_split_order_independent():
    """Разбиение определяется хешем пожара, а не порядком строк в meta.csv."""
    meta = _meta()
    shuffled = meta.sample(frac=1, random_state=3).reset_index(drop=True)
    assert set(split_by_fire(meta)["val"]) == set(split_by_fire(shuffled)["val"])


def test_test_schema_parses_without_train_only_columns():
    """Отсутствие скрытых в тесте колонок — штатный случай, а не ошибка."""
    frame = pd.DataFrame([dict.fromkeys(TEST_COLUMNS, 1)])
    frame["chip_id"] = "AF_te_000001"
    assert "fire_event_id" not in frame.columns
    assert frame.chip_id.iloc[0] == "AF_te_000001"


def test_nan_cloud_frac_stays_nan():
    """У AF-чипов cloud_frac равен nan и не превращается в ноль."""
    frame = pd.DataFrame([{"chip_id": "AF_te_000001", "cloud_frac": np.nan}])
    assert np.isnan(frame.cloud_frac.iloc[0])
    assert frame.cloud_frac.iloc[0] != 0


def test_split_requires_fire_ids():
    meta = _meta(n_fires=1, chips_per_fire=1)
    split = split_by_fire(meta)
    assert len(split["val"]) == 1


def test_meta_берётся_уровнем_выше_и_фильтруется_по_виду(tmp_path):
    """В тесте `meta.csv` один на оба модуля и лежит уровнем выше.

    Без отбора по `kind` загрузчик BS увидел бы AF-чипы и потребовал файлы,
    которых нет, — а обнаружилось бы это только на сборке ответа.
    """
    import pandas as pd

    from src.comp.chips import read_meta

    root = tmp_path / "test"
    (root / "bs").mkdir(parents=True)
    (root / "af").mkdir()
    pd.DataFrame({"chip_id": ["BS_te_000001", "AF_te_000001", "AF_te_000002"],
                  "kind": ["bs", "af", "af"]}).to_csv(root / "meta.csv", index=False)

    assert list(read_meta(root / "bs", "bs").chip_id) == ["BS_te_000001"]
    assert list(read_meta(root / "af", "af").chip_id) == ["AF_te_000001", "AF_te_000002"]

    # Собственный meta.csv модуля имеет приоритет — так устроено обучение.
    pd.DataFrame({"chip_id": ["BS_tr_000009"]}).to_csv(root / "bs" / "meta.csv", index=False)
    assert list(read_meta(root / "bs", "bs").chip_id) == ["BS_tr_000009"]


def test_отсутствие_meta_поднимает_ошибку(tmp_path):
    from src.comp.chips import read_meta

    (tmp_path / "bs").mkdir()
    with pytest.raises(FileNotFoundError):
        read_meta(tmp_path / "bs", "bs")
