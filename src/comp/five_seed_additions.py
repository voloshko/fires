"""SPEC-118: explicit parent/new member identity for one uniform five-seed ensemble."""
import json
from pathlib import Path
from argparse import Namespace

import torch

from .accuracy_artifacts import checkpoint_metadata
from .accuracy_contract import canonical_hash, sha256_file
from .equal_cost_additions import LOSS_ID
from .temporal_adapter_contract import BASE_SHAS, fit_population, read_seal

FIVE_BASE_SHAS = (*BASE_SHAS,
    '504141c858b50615e0a884dfa512f20360ceb2b86455967cb056279d2497ba8b',
    'a48abe8b5c98883cf2f60a23eb457e39e5fea7584bafb4266d05eb7bcb16719f')
PARENT_ROOT = Path('research/equal-cost-additions-v1')
PARENT_RESULT_SHA = 'eddf5ba5df6e6d47265503234a7443d2ee4730df9c3c1eef25eefb30105f51bc'
PARENT_RECIPE = 'a6e432e3f0eeb21ac550027e7b5278d67aa1639ed819ed40c4d53539484ab690'


def validate_member(bundle, seed, variant, recipe_id, allow_smoke=False):
    if seed not in (1, 2, 3, 4, 5) or variant not in ('pair', 'post'):
        raise ValueError('fixed five-member identity required')
    family, recipe = ('SPEC-114', PARENT_RECIPE) if seed <= 3 else ('SPEC-118', recipe_id)
    if (bundle.get('spec_id') != family or bundle.get('loss_id') != LOSS_ID
            or bundle.get('base_sha256') != FIVE_BASE_SHAS[seed-1]
            or bundle.get('seed') != seed or bundle.get('variant') != variant
            or bundle.get('rank') != 8 or bundle.get('recipe_id') != recipe
            or (bundle.get('smoke') is not False and not (allow_smoke and bundle.get('smoke') is True))):
        raise ValueError('five-member family/seed/loss/base/recipe binding differs')


def contribution(addition, members):
    if members != 5 or not torch.isfinite(addition).all() or (addition < 0).any() or (addition > 1).any():
        raise ValueError('five finite addition probabilities required')
    return addition.float()/5


def member_paths(runs, variant):
    if variant not in ('pair', 'post'):
        raise ValueError('fixed pair/post variant required')
    return [PARENT_ROOT/f'{variant}-s{s}/adapter.pt' for s in (1, 2, 3)]+[
        Path(runs)/f'{variant}-s{s}/adapter.pt' for s in (4, 5)]


def audit_parent(data_root):
    from scripts.score_equal_cost_additions import audit_models
    if sha256_file(PARENT_ROOT/'development.json') != PARENT_RESULT_SHA:
        raise ValueError('immutable parent result changed')
    seal = read_seal(PARENT_ROOT/'recipe.json')
    if seal['recipe_id'] != PARENT_RECIPE:
        raise ValueError('immutable parent training recipe changed')
    base = json.loads((Path(data_root)/'forest-pairs/manifest.json').read_text())['records']
    russian = json.loads((Path(data_root)/'russia/manifest.json').read_text())['records']
    fit = fit_population(base, russian)
    paths, metadata, summaries = audit_models(Namespace(runs=PARENT_ROOT), fit, {}, seal)
    parent = json.loads((PARENT_ROOT/'development.json').read_text())
    scoring = json.loads((PARENT_ROOT/'development.recipe.json').read_text())
    if (scoring['recipe_id'] != parent['scoring_recipe_id']
            or canonical_hash({k: v for k, v in scoring.items() if k != 'recipe_id'}) != scoring['recipe_id']):
        raise ValueError('immutable parent scoring recipe changed')
    for name, digest in scoring['source_hashes'].items():
        if sha256_file(name) != digest:
            raise ValueError('immutable parent scoring source changed')
    hashes = {v: [sha256_file(p) for p in ps] for v, ps in paths.items()}
    if hashes != parent['adapter_hashes']:
        raise ValueError('parent adapter bytes changed')
    for variant in ('pair', 'post'):
        for i, value in enumerate(metadata[variant], 1):
            validate_member(value, i, variant, PARENT_RECIPE)
    return paths, metadata, summaries, hashes


def audit_all(args, fit, populations, seal):
    parent_paths, parent_metadata, parent_summaries, hashes = audit_parent(args.data_root)
    if seal['configuration']['parent_adapters'] != hashes:
        raise ValueError('new training recipe changed parent membership')
    expected_ids = [r['sample_id'] for r in fit]
    excluded = {r['sample_id'] for rows in populations.values() for r in rows}
    if set(expected_ids) & excluded:
        raise ValueError('evaluation population was fitted')
    expected_exposure = {r['sample_id']: 30*int(r['sampling_weight']) for r in fit}
    paths, metadata, summaries = {}, {}, {}
    for variant in ('pair', 'post'):
        paths[variant] = member_paths(args.runs, variant)
        metadata[variant], summaries[variant] = list(parent_metadata[variant]), list(parent_summaries[variant])
        for seed, path in zip((4, 5), paths[variant][3:]):
            m = checkpoint_metadata(path)
            validate_member(m, seed, variant, seal['recipe_id'])
            if (m['fit_ids'] != expected_ids or m['epochs'] != 30 or m['batch'] != 8
                    or m['source_hashes'] != seal['source_hashes']
                    or m['base_manifest'] != seal['configuration']['base_manifest']
                    or m['russian_manifest'] != seal['configuration']['russian_manifest']):
                raise ValueError('new member source/fit/budget differs')
            s = json.loads(path.with_name('summary.json').read_text())
            if (s.get('completed') is not True or s.get('backbone_unchanged') is not True
                    or s['actual_exposures'] != expected_exposure or len(s['losses']) != 30
                    or s['recipe_id'] != seal['recipe_id']):
                raise ValueError('new full member execution incomplete')
            metadata[variant].append(m); summaries[variant].append(s)
    for i in range(5):
        a, b = metadata['pair'][i], metadata['post'][i]
        if a['mean'] != b['mean'] or a['std'] != b['std']:
            raise ValueError('matched member normalization differs')
        for key in ('epoch_order_hashes', 'base_tensor_digest', 'actual_exposures'):
            if summaries['pair'][i][key] != summaries['post'][i][key]:
                raise ValueError('matched member batches/initializer differ')
    return paths, metadata, summaries
