"""SPEC-119: successor identity guard around immutable113 storage contracts."""
from .eastern_confirmation import (
    DATA_BUDGET, FREE_FLOOR, RESERVATION, SELECTION_SHA, StorageBudget,
    cache_guard, confirmation_decision, storage_guard,
)

TRAINING_RECIPE = 'ccf1df7dd1eb779f7b2d5d66019751eb675b1b48e916716c3171837692f550dc'


def require_eligible(candidate):
    expected = {'pooled_gain', 'regional_points', 'mediterranean', 'missed_groups'}
    if (candidate.get('spec_id') != 'SPEC-118' or candidate.get('eligible') is not True
            or set(candidate.get('guards', {})) != expected
            or any(value is not True for value in candidate['guards'].values())):
        raise ValueError('complete eligible SPEC118 candidate required before external I/O')


def require_admission(manifest):
    if (manifest.get('spec_id') != 'SPEC-119' or manifest.get('ready_for_evaluation') is not True
            or manifest.get('qc_probe') is not False or manifest.get('selection_sha256') != SELECTION_SHA
            or manifest.get('counts', {}).get('samples', 0) < 50
            or manifest.get('counts', {}).get('groups', 0) < 20):
        raise ValueError('actual50windows/20groups SPEC116 source admission required')
