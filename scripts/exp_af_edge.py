"""SPEC-40: fixed AF edge expansion on frozen v15 OOF probabilities."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.af import AfDataset
from src.comp.hypothesis_lab import binary_counts, binary_scores, digest, paired_interval, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--probabilities', default='/tmp/af_oof_valid.npy')
    args = parser.parse_args()
    root = Path('research/af-hard-v2')
    chips = sorted(p.stem for p in (root / 'hard_probabilities').glob('*.npz'))
    if len(chips) != 336:
        raise ValueError('expected all 336 OOF chips')
    probabilities = np.load(args.probabilities, mmap_mode='r')
    if probabilities.size != len(chips) * 256 * 256:
        raise ValueError('unexpected probability size')
    probabilities = probabilities.reshape(len(chips), 256, 256)
    dataset = AfDataset('data/comp/train/af')
    hashes = {args.probabilities: digest(args.probabilities), __file__: digest(__file__)}
    rows = {'baseline': [], 'expanded': []}
    for i, chip_id in enumerate(chips):
        cache = root / 'cache' / f'{chip_id}.npz'
        z = np.load(cache)
        chip = dataset.load(chip_id)
        truth, valid = chip.mask > 0, chip.valid()
        if not np.array_equal(z['y'].reshape(chip.shape), truth) or not np.array_equal(z['valid'].reshape(chip.shape), valid):
            raise ValueError('cache/source truth mismatch')
        hashes[str(cache)] = digest(cache)
        for folder, suffix in [('viirs', 'VIIRS_I1-I5'), ('aux', 'AUX'), ('masks', 'MASK')]:
            path = Path('data/comp/train/af') / folder / f'{chip_id}_{suffix}.tif'
            hashes[str(path)] = digest(path)
        p = probabilities[i]
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('invalid probability')
        baseline = (p >= .45) & valid
        near_strong = maximum_filter((p >= .75) & valid, size=3, mode='constant', cval=0)
        expanded = baseline | (near_strong & (p >= .35) & valid)
        rows['baseline'].append(binary_counts(truth, baseline))
        rows['expanded'].append(binary_counts(truth, expanded))
    counts = {k: np.asarray(v) for k, v in rows.items()}
    metrics = {k: binary_scores(v) for k, v in counts.items()}
    if abs(metrics['baseline']['f1'] - .9515) > .0001:
        raise ValueError('v15 OOF baseline did not reproduce')
    improvement = metrics['expanded']['f1'] - metrics['baseline']['f1']
    ci = paired_interval(counts['baseline'], counts['expanded'])
    halves = {}
    for seed in (1, 17, 42):
        order = np.random.default_rng(seed).permutation(len(chips))
        halves[str(seed)] = [binary_scores(counts['expanded'][part])['f1'] - binary_scores(counts['baseline'][part])['f1'] for part in np.array_split(order, 2)]
    report = dict(metrics=metrics, delta=improvement, ci95_chip_bootstrap=ci, half_deltas=halves,
                  accepted=bool(improvement >= .005 and ci[0] > 0 and all(x > 0 for values in halves.values() for x in values)),
                  chips=chips, per_chip={k: v.tolist() for k, v in counts.items()}, source_sha256=hashes,
                  non_claims=['Pooled OOF baseline threshold was selected on these labels; not nested evaluation.',
                      'Chip bootstrap does not establish event independence or hidden-test quality.',
                      'Neighbor-provided OOF array was hashed and baseline replayed, not independently retrained.'])
    write_json(args.out, report)
    print({k: v for k, v in report.items() if k not in ('per_chip', 'source_sha256', 'chips')})


if __name__ == '__main__':
    main()
