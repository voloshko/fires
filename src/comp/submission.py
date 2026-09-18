"""Сборка и валидация submission.csv (SPEC-16).

Валидатор воспроизводит правила из постановки, а не проверяющую систему
организаторов: расхождение между ними возможно. Молча собранный неверный файл —
худший исход, поэтому любое нарушение даёт ненулевой код возврата.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .rle import decode, encode

COLUMNS = ("chip_id", "class_id", "rle")
BS_CLASSES = (1, 2, 3)
AF_CLASSES = (1,)


def rows_for_chip(chip_id: str, mask: np.ndarray, classes: tuple[int, ...]) -> list[dict]:
    """Маска классов (0 = фон) -> строки сабмита, по одной на класс."""
    return [
        {"chip_id": chip_id, "class_id": cls, "rle": encode(mask == cls)}
        for cls in classes
    ]


def write(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def validate(
    file: str | Path,
    template: str | Path | None = None,
    shapes: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Возвращает список нарушений; пустой список — файл валиден."""
    faults: list[str] = []
    frame = pd.read_csv(file, dtype={"chip_id": str, "rle": str}, keep_default_na=False)

    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        return [f"нет колонок: {', '.join(missing)}"]

    if (frame.chip_id == "").any() or (frame.class_id.astype(str) == "").any():
        faults.append("пустые ячейки в chip_id или class_id")
    if frame.rle.isna().any() or frame.rle.astype(str).str.lower().eq("nan").any():
        faults.append("NaN в колонке rle")

    pairs = set(zip(frame.chip_id, frame.class_id.astype(int)))
    if len(pairs) != len(frame):
        faults.append("дублирующиеся пары (chip_id, class_id)")

    if template is not None:
        want = pd.read_csv(template, dtype={"chip_id": str})
        expected = set(zip(want.chip_id, want.class_id.astype(int)))
        if extra := pairs - expected:
            faults.append(f"лишние пары: {sorted(extra)[:3]}… всего {len(extra)}")
        if lost := expected - pairs:
            faults.append(f"недостающие пары: {sorted(lost)[:3]}… всего {len(lost)}")

    if shapes:
        for chip_id, group in frame.groupby("chip_id"):
            shape = shapes.get(chip_id)
            if shape is None:
                faults.append(f"{chip_id}: размер чипа неизвестен")
                continue
            union = np.zeros(shape, dtype=bool)
            for _, row in group.iterrows():
                try:
                    mask = decode(str(row.rle), shape)
                except ValueError as exc:
                    faults.append(f"{chip_id} класс {row.class_id}: {exc}")
                    continue
                if (union & mask).any():
                    faults.append(f"{chip_id}: классы пересекаются внутри чипа")
                union |= mask
    return faults


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m src.comp.submission")
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--file", required=True)
    check.add_argument("--template")
    args = parser.parse_args(argv)

    faults = validate(args.file, args.template)
    if faults:
        for fault in faults:
            print(f"НАРУШЕНИЕ: {fault}", file=sys.stderr)
        return 1
    rows = len(pd.read_csv(args.file))
    print(f"OK: {args.file}, строк без заголовка: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
