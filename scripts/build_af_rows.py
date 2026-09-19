"""Пересобрать только AF-строки сабмита моделью из models/af_hgb.pkl,
BS-строки скопировать байт-в-байт из базового файла (SPEC-19/38)."""
import sys, csv, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.comp.af import AfDataset, predict
from src.comp.rle import encode
from src.comp.submission import validate
from scripts.build_af_hard_candidate import replace_af_rows
p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--test', required=True); p.add_argument('--out', required=True); a = p.parse_args()
d = AfDataset(Path(a.test) / 'af'); rows = {}; px = 0; empty = 0
for c in d.chip_ids():
    m = predict(d.load(c)); rows[c] = encode(m > 0); px += int(m.sum()); empty += int(m.sum() == 0)
info = replace_af_rows(a.base, rows, a.out)
faults = validate(a.out, Path(a.test) / 'sample_submission.csv')
print(f'AF пикселей {px}, пустых чипов {empty}/{len(rows)}, изменённых строк {info["changed_af_rows"]}, BS сохранён: {info["bs_bytes_preserved"]}, валидатор: {faults or "OK"}')
