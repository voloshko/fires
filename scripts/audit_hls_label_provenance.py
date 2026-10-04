"""SPEC-115: source-only archival/reuse diagnostics for all17 unresolved fit scenes."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comp.accuracy_contract import sha256_file
from src.comp.accuracy_artifacts import execution_sources, freeze_recipe
from src.comp.hls_label_provenance import label_signature, prior_ignition, reuse_witness

REGISTRY_SHA = '86f2210603e5921d384c62a52e90167ae64abbad3473bc1cce686ea7bbad3ed2'


def read_target(root, record):
    image = root/record['source']
    mask = image.with_name(image.name.replace('_merged.tif', '.mask.tif'))
    if sha256_file(image) != record['source_sha256'] or sha256_file(mask) != record['label_sha256']:
        raise ValueError('source/label bytes differ from immutable registry')
    with rasterio.open(image) as source, rasterio.open(mask) as labels:
        if (source.crs is None or source.crs != labels.crs or source.shape != labels.shape
                or source.transform != labels.transform):
            raise ValueError('source and label grid differ')
        values, target = source.read(), labels.read(1)
        valid = (values != -9999).all(0) & np.isfinite(values).all(0) & (target >= 0)
        grid = dict(crs=str(source.crs), affine=list(source.transform)[:6])
        metadata = dict(image_tags=source.tags(), mask_tags=labels.tags(),
                        image_descriptions=list(source.descriptions), mask_descriptions=list(labels.descriptions))
        return target, valid, grid, metadata


def filter_prior_candidates(candidates, acquisition):
    selected = candidates.audit_day.map(lambda d: prior_ignition(d, acquisition)).astype(bool)
    # An empty untyped Series used with [] selects columns and drops geometry.
    return candidates.loc[selected].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-root', 'registry', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or sha256_file(args.registry) != REGISTRY_SHA:
        raise ValueError('immutable output or registry mismatch')
    registry = json.loads(args.registry.read_text())
    records = registry['records']
    unresolved = [r for r in records if r['decision'] == 'unknown']
    if len(records) != 540 or len(unresolved) != 17 or len({r['sample_id'] for r in records}) != 540:
        raise ValueError('original population incomplete')
    vector = args.source_root/'external/mtbs/mtbs_perims_DD.shp'
    vector_hashes = {p.name: sha256_file(p) for p in sorted(vector.parent.glob('mtbs_perims_DD.*')) if p.is_file()}
    if vector_hashes != registry['vector_hashes']:
        raise ValueError('official vector changed')
    publisher = args.source_root/'external/hls_burn_scars'
    metadata_paths = [publisher/'README.md', publisher/'training/training_index.csv',
                      publisher/'validation/validation_index.csv']
    hashes = {str(p.relative_to(args.source_root)): sha256_file(p) for p in metadata_paths}
    index_summary = {}
    for path in metadata_paths[1:]:
        with path.open(newline='') as stream:
            rows = list(csv.reader(stream))
        widths = sorted({len(row) for row in rows})
        if widths != [2] or any(not r[0].endswith('_merged.tif') or not r[1].endswith('.mask.tif') for r in rows):
            raise ValueError('publisher index schema differs; inspect explicitly')
        index_summary[path.name] = dict(rows=len(rows), columns=['image_filename', 'mask_filename'],
                                      publisher_event_id_available=False)
    if shutil.disk_usage(args.out.parent).free < 3*1024**3+8*1024**2:
        raise ValueError('audit free-space floor unavailable before sealing')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recipe = freeze_recipe(args.out.with_suffix('.recipe.json'), execution_sources(Path(__file__),
        'src/comp/hls_label_provenance.py'), dict(registry_sha256=REGISTRY_SHA,
        vector_hashes=vector_hashes, publisher_metadata_hashes=hashes,
        ordered_unresolved_ids=[r['sample_id'] for r in unresolved], scope='fit-only; all prior ignitions; no event assignment'))
    signatures, inputs = {}, {}
    for record in records:
        target, valid, grid, metadata = read_target(args.source_root, record)
        signature = label_signature(target, valid, grid)
        signatures.setdefault(signature, []).append(record)
        if record['decision'] == 'unknown':
            inputs[record['sample_id']] = (target, valid, grid, metadata, signature)
    frame = gpd.read_file(vector).to_crs(4326)
    frame['audit_ignition'] = pd.to_datetime(frame['ig_date'], utc=True, errors='coerce')
    rows = []
    for record in unresolved:
        target, valid, grid, metadata, signature = inputs[record['sample_id']]
        positive = (target == 1) & valid
        candidates = frame.iloc[frame.sindex.query(box(*record['bbox_wgs84']), predicate='intersects')].copy()
        unknown_dates = int(candidates.audit_ignition.isna().sum())
        candidates['audit_day'] = candidates.audit_ignition.map(lambda d: None if pd.isna(d) else d.date().isoformat())
        candidates = filter_prior_candidates(candidates, record['acquisition_day'])
        scores = []
        for event, group in candidates.to_crs(grid['crs']).groupby('event_id'):
            dates = sorted(set(group.audit_day))
            pixels = rasterize([(group.geometry.union_all(), 1)], out_shape=target.shape,
                transform=rasterio.Affine(*grid['affine']), fill=0, dtype='uint8').astype(bool)
            hits, total = int((pixels & positive).sum()), int(positive.sum())
            age = [(pd.Timestamp(record['acquisition_day'])-pd.Timestamp(d)).days for d in dates]
            scores.append(dict(event_id=str(event), ignition_days=dates, ages_days=age,
                               overlap_positive=hits, positive_pixels=total, coverage=hits/total if total else None))
        scores.sort(key=lambda r: (-(r['coverage'] or 0), r['event_id']))
        reuse = [reuse_witness(r) for r in signatures[signature] if r['sample_id'] != record['sample_id']]
        rows.append(dict(sample_id=record['sample_id'], acquisition_day=record['acquisition_day'],
            source_sha256=record['source_sha256'], label_sha256=record['label_sha256'],
            signature=signature, metadata=metadata, positive_pixels=int(positive.sum()),
            identical_label_domain_witnesses=reuse, unknown_ignition_candidates=unknown_dates,
            prior_ignition_candidates=scores, event_assignment=None))
    result = dict(spec_id='SPEC-115', registry_sha256=REGISTRY_SHA, recipe_id=recipe,
        publisher_metadata_hashes=hashes, indices=index_summary, records=rows,
        summary=dict(unresolved=len(rows), exact_reuse_windows=sum(bool(r['identical_label_domain_witnesses']) for r in rows),
            older_95pct_candidate_windows=sum(any(c['coverage'] is not None and c['coverage'] >= .95
                and min(c['ages_days']) > 450 for c in r['prior_ignition_candidates']) for r in rows),
            absent_prior_candidate_windows=sum(not r['prior_ignition_candidates'] for r in rows)),
        non_claims=['No publisher lookup, new event assignment or physical ignition inference.',
            'No change to SPEC96/97/93 source admission, protected pixel retention, model metrics or full goal completion.'])
    content = json.dumps(result, indent=2).encode()
    if len(content) > 8*1024**2 or shutil.disk_usage(args.out.parent).free-len(content) < 3*1024**3:
        raise ValueError('audit output budget/free-space floor unavailable')
    with args.out.open('xb') as stream:
        stream.write(content)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
