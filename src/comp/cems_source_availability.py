"""SPEC-117: country-tagged activation availability, never physical fire identity."""
from datetime import datetime

RUSSIA = {'russia', 'russian federation', 'россия', 'российская федерация'}
BOREAL = {'canada', 'sweden', 'finland', 'norway'}


def normalize(record):
    code = record.get('code')
    if not isinstance(code, str) or not code:
        raise ValueError('activation code unavailable')
    countries = record.get('countries')
    if countries is None:
        countries = []
    if not isinstance(countries, list):
        raise ValueError('malformed country schema')
    names, unknown_country = [], not countries
    for country in countries:
        name = country.get('short_name') if isinstance(country, dict) else country
        if name is None or (isinstance(name, str) and not name.strip()):
            unknown_country = True
            continue
        if not isinstance(name, str):
            raise ValueError('malformed country label')
        names.append(name.strip())
    category = record.get('category')
    if isinstance(category, dict):
        slug, name = category.get('slug'), category.get('name')
        if not isinstance(slug, str) or not slug or not isinstance(name, str) or not name:
            raise ValueError('malformed category schema')
        wildfire = slug.casefold() == 'fire'
    elif isinstance(category, str) and category:
        name = category
        wildfire = name.casefold() == 'wildfire'
    else:
        raise ValueError('malformed category schema')
    timestamp = record.get('activationTime')
    if not isinstance(timestamp, str):
        raise ValueError('activation timestamp unavailable')
    datetime.fromisoformat(timestamp)
    return dict(code=code, countries=names, category=name, wildfire=wildfire,
        activation_time=timestamp, timestamp_semantics='activation, not physical ignition',
        name=record.get('name'), n_aois=record.get('n_aois'), n_products=record.get('n_products'),
        country_unknown=unknown_country)


def summarize(records, declared_count):
    if type(declared_count) is not int or declared_count < 0 or len(records) != declared_count:
        raise ValueError('incomplete declared listing')
    rows = [normalize(r) for r in records]
    if len({r['code'] for r in rows}) != len(rows):
        raise ValueError('duplicate activation codes')
    russian = [r for r in rows if any(c.casefold() in RUSSIA for c in r['countries'])]
    boreal = [r for r in rows if r['wildfire'] and any(c.casefold() in BOREAL for c in r['countries'])]
    return dict(records=len(rows), unknown_country_records=[r for r in rows if r['country_unknown']],
        declared_russian_records=russian, declared_russian_wildfires=[r for r in russian if r['wildfire']],
        boreal_wildfire_activations=boreal,
        post2015_boreal_activations=[r for r in boreal if r['activation_time'] >= '2015-01-01'],
        activation_time_bounds=[min((r['activation_time'] for r in rows), default=None),
                                max((r['activation_time'] for r in rows), default=None)])
