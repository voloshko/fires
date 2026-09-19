"""SPEC-39: one frozen robust-pooling alternative, CPU cache replay only."""
import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_unet import load_split
from src.comp.hypothesis_lab import (
    bs_prediction, bs_scores, confusion, digest, verify_bs_probability_cache, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    dataset, fit, evaluation = load_split('data/comp/train/bs', 'data/comp/split_bs.json')
    hashes = verify_bs_probability_cache('research/bs-boost-v1', fit, evaluation)
    hashes[__file__] = digest(__file__)
    boost = np.load('research/bs-boost-v1/probabilities.npy', mmap_mode='r')
    arrays = []
    for tag in ('d7opt', 'd7opt_s1', 'd7opt_s2', 'd7optjit', 'd7optjit_s1'):
        path = Path('baseline_models') / f'exp_{tag}.tune.npy'
        hashes[str(path)] = digest(path)
        array = np.load(path, mmap_mode='r')
        if array.shape != boost.shape:
            raise ValueError('cache shape mismatch')
        arrays.append(array)
    counts = {'mean': [], 'trimmed': []}
    clear = {'mean': [], 'trimmed': []}
    for i, chip_id in enumerate(evaluation):
        chip = dataset.load(chip_id)
        probabilities = np.stack([a[i].astype(np.float32) for a in arrays])
        if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
            raise ValueError('invalid probabilities')
        if not np.allclose(probabilities.sum(-1), 1, atol=.002):
            raise ValueError('cache class dimension or probability sum mismatch')
        ordinary = probabilities.mean(0)
        trimmed = np.sort(probabilities, axis=0)[1:-1].mean(0)
        total = trimmed.sum(-1, keepdims=True)
        if (total <= 0).any():
            raise ValueError('zero trimmed probability mass')
        trimmed /= total
        for name, probability in (('mean', ordinary), ('trimmed', trimmed)):
            prediction = bs_prediction(probability, boost[i].astype(np.float32), chip)
            counts[name].append(confusion(chip.mask, prediction))
            clear[name].append(confusion(chip.mask, prediction, valid=chip.valid()))
    counts = {k: np.asarray(v) for k, v in counts.items()}
    metrics = {k: bs_scores(v.sum(0)) for k, v in counts.items()}
    if abs(metrics['mean']['weighted'] - .744275665) > 1e-6:
        raise ValueError('v15 BS control did not reproduce')
    def delta(indices):
        return bs_scores(counts['trimmed'][indices].sum(0))['weighted'] - bs_scores(counts['mean'][indices].sum(0))['weighted']
    rng = np.random.default_rng(20260918)
    ci = np.quantile([delta(rng.integers(len(evaluation), size=len(evaluation))) for _ in range(2000)], [.025, .975])
    halves = {}
    for seed in (1, 17, 42):
        order = np.random.default_rng(seed).permutation(len(evaluation))
        halves[str(seed)] = [delta(part) for part in np.array_split(order, 2)]
    improvement = metrics['trimmed']['weighted'] - metrics['mean']['weighted']
    report = dict(metrics=metrics, delta=improvement, ci95_chip_bootstrap=ci.tolist(), half_deltas=halves,
                  accepted=bool(improvement >= .005 and ci[0] > 0 and all(x > 0 for values in halves.values() for x in values)),
                  clear_sky={k: bs_scores(np.sum(v, axis=0)) for k, v in clear.items()},
                  evaluation=evaluation, fit=fit, per_chip={k: v.tolist() for k, v in counts.items()},
                  source_sha256=hashes, non_claims=['Adaptive development hypothesis, not an independent test.',
                      'Does not measure final all224 model accuracy or closed-test improvement.'])
    write_json(args.out, report)
    print({k: v for k, v in report.items() if k not in ('per_chip', 'source_sha256', 'fit', 'evaluation')})


if __name__ == '__main__':
    main()
