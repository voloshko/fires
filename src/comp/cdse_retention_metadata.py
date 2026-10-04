"""SPEC-120: exact frozen loss identities and metadata-only L2A descriptors."""
from datetime import datetime, timedelta
import re
import uuid

MANIFEST_SHA = '75d1afa37b8dd51d98ea23c2d364b6f1f24e6c253e8ed354f7167d2dcfe3794b'
ENDPOINT = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'


def lost_population(manifest):
    bound = manifest['conservative_event_bound']
    rows = bound['lost_protected']
    if (manifest.get('spec_id') != 'SPEC-93' or manifest.get('qc_probe') is not False
            or manifest.get('ready_for_training') is not False
            or bound.get('original_known_events') != 886 or bound.get('retained_known_events') != 861
            or len(rows) != 16 or len({r['sample_id'] for r in rows}) != 16
            or any(type(r['lost_protected_pixels']) is not int or r['lost_protected_pixels'] <= 0 for r in rows)
            or sum(r['lost_protected_pixels'] for r in rows) != 2784):
        raise ValueError('fixed full16window/2784pixel population required')
    return rows


def identity(sample):
    match = re.search(r'HLS\.S30\.(T\d{2}[A-Z]{3})\.(\d{7})\.', sample)
    if match is None:
        raise ValueError('frozen HLS day/tile unavailable')
    day = datetime.strptime(match[2], '%Y%j')
    if day.strftime('%Y%j') != match[2]:
        raise ValueError('invalid HLS acquisition day')
    return match[1], day.date().isoformat()


def query(sample):
    tile, day = identity(sample)
    following = (datetime.fromisoformat(day)+timedelta(days=1)).date().isoformat()
    expression = ("Collection/Name eq 'SENTINEL-2' and contains(Name,'_MSIL2A_') "
        f"and contains(Name,'_{tile}_') and ContentDate/Start ge {day}T00:00:00.000Z "
        f"and ContentDate/Start lt {following}T00:00:00.000Z")
    return dict(filter=expression, tile=tile, acquisition_day=day)


def descriptor(product, tile, day):
    name = product.get('Name')
    match = re.fullmatch(r'S2[ABCD]_MSIL2A_(\d{8})T\d{6}_N(\d{4})_R\d{3}_(T\d{2}[A-Z]{3})_\d{8}T\d{6}(?:\.SAFE)?', name or '')
    if match is None or match[3] != tile or datetime.strptime(match[1], '%Y%m%d').date().isoformat() != day:
        raise ValueError('catalogue product sensingday/tile differs')
    source_day = datetime.fromisoformat(product['ContentDate']['Start'].replace('Z', '+00:00')).date().isoformat()
    if source_day != day:
        raise ValueError('catalogue ContentDate differs')
    identifier = str(uuid.UUID(product['Id']))
    online = product.get('Online')
    return dict(id=identifier, name=name, tile=tile, acquisition_day=day,
        publication_date=product.get('PublicationDate'), processing_baseline_token='N'+match[2],
        processing_baseline_known=match[2] not in ('0000', '9999'), encoding_verified=False,
        online_state='online' if online is True else 'offline' if online is False else 'unknown',
        pixel_admission='NOT_TESTED')


def descriptors(products, count, tile, day):
    if type(count) is not int or count < 0 or len(products) != count:
        raise ValueError('incomplete public product listing')
    rows = [descriptor(p, tile, day) for p in products]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate product IDs')
    return rows
