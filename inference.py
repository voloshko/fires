#!/usr/bin/env python3
"""Точка входа инференса (SPEC-20).

Модуль физически не имеет доступа к сети: он читает только каталог с чипами и
обученную модель с диска. Ни один ответ по тестовому чипу не получен из FIRMS,
VNP14 или MOD14 — применение этих продуктов запрещено правилами кейса, и
запрет обеспечен устройством модуля, а не обещанием.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.comp.baseline import predict as predict_threshold
from src.comp.chips import BsDataset
from src.comp.model import load as load_model
from src.comp.model import predict as predict_model
from src.comp.ensemble import NET_WEIGHT as ENSEMBLE_WEIGHT
from src.comp.submission import BS_CLASSES, rows_for_chip, write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inference.py")
    parser.add_argument("--data-dir", required=True, help="каталог теста (bs/ и af/)")
    parser.add_argument("--model", default="models/bs_hgb.pkl")
    parser.add_argument("--unet", nargs="+", default=["models/bs_unet.pt"],
                        help="сети по гари (несколько — сидовый ансамбль, вероятности "
                             "усредняются); при отсутствии всех работает бустинг")
    parser.add_argument("--unet-weights", nargs="+", type=float,
                        help="веса сетей в порядке --unet (по умолчанию равные)")
    parser.add_argument("--net-weight", type=float, default=ENSEMBLE_WEIGHT,
                        help="доля сети в ансамбле; 1.0 — только сеть, 0.0 — только бустинг")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--template", help="sample_submission.csv для проверки состава")
    args = parser.parse_args(argv)

    started = time.time()
    rows: list[dict] = []

    bs_root = Path(args.data_dir) / "bs"
    if bs_root.exists():
        dataset = BsDataset(bs_root)
        # Приоритет: сеть, затем бустинг, затем порог. О каждом понижении
        # сообщаем в stderr — молча сработавший запасной вариант выглядит как
        # успешный прогон и портит выводы о качестве.
        from src.comp import ensemble, unet

        net = None
        found = [p for p in args.unet if Path(p).exists()]
        weights = None
        if args.unet_weights:
            if len(args.unet_weights) != len(args.unet):
                parser.error("--unet-weights должен совпадать по длине с --unet")
            weights = [w for p, w in zip(args.unet, args.unet_weights) if Path(p).exists()]
        if found and unet.available():
            net = [unet.load(p) for p in found]
        elif found:
            print("сеть есть, но torch недоступен — беру бустинг", file=sys.stderr)

        model = load_model(args.model) if Path(args.model).exists() else None

        if net is not None and model is not None:
            print(f"гарь: ансамбль {len(net)} сетей {found} и {args.model}, вес сети {args.net_weight}, веса сетей {weights or 'равные'}")
        elif net is not None:
            print(f"гарь: только сети {found} — бустинга на диске нет", file=sys.stderr)
        elif model is not None:
            print(f"гарь: только бустинг {args.model} — сети на диске нет", file=sys.stderr)
        else:
            print("ни сети, ни модели — падаю на пороговый baseline", file=sys.stderr)

        for chip_id in dataset.chip_ids():
            chip = dataset.load(chip_id)
            if net is not None and model is not None:
                mask = ensemble.predict(net, model, chip, args.net_weight, net_weights=weights)
            elif net is not None:
                mask = unet.predict(net[0], chip)
            elif model is not None:
                mask = predict_model(model, chip)
            else:
                mask = predict_threshold(chip)
            rows.extend(rows_for_chip(chip_id, mask, BS_CLASSES))
        print(f"BS: {len(dataset)} чипов")

    af_root = Path(args.data_dir) / "af"
    if af_root.exists():
        from src.comp.af import AfDataset
        from src.comp.af import predict as predict_af

        af = AfDataset(af_root)
        for chip_id in af.chip_ids():
            mask = predict_af(af.load(chip_id))
            rows.extend(rows_for_chip(chip_id, mask, (1,)))
        print(f"AF: {len(af)} чипов")

    write(rows, args.out)
    elapsed = time.time() - started
    print(f"{args.out}: строк {len(rows)}, время {elapsed:.1f}с")

    if args.template:
        from src.comp.submission import validate

        faults = validate(args.out, args.template)
        for fault in faults:
            print(f"НАРУШЕНИЕ: {fault}", file=sys.stderr)
        return 1 if faults else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
