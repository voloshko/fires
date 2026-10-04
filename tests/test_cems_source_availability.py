import pytest
from src.comp.cems_source_availability import normalize, summarize


def row(**values):
    return dict(code='EMSR1', countries=['Sweden'], category='Wildfire',
                activationTime='2018-07-16T21:17:00', **values)


def test_public_and_historical_schema_match():
    current = row()
    historic = {**current, 'countries': [{'short_name': 'Sweden'}],
                'category': {'slug': 'fire', 'name': 'Wildfire'}}
    assert normalize(current) == normalize(historic)
    assert normalize(current)['timestamp_semantics'] == 'activation, not physical ignition'


def test_country_unknown_is_preserved_and_not_inferred_from_name():
    unknown = {**row(), 'countries': None, 'name': 'Russian fire'}
    result = summarize([unknown], 1)
    assert len(result['unknown_country_records']) == 1
    assert result['declared_russian_records'] == []


def test_declared_russia_and_old_boreal_activation_counts():
    russian = {**row(), 'countries': ['Russian Federation']}
    old = {**row(), 'code': 'EMSR2', 'activationTime': '2014-08-04T13:59:00'}
    result = summarize([russian, old], 2)
    assert len(result['declared_russian_wildfires']) == 1
    assert len(result['boreal_wildfire_activations']) == 1
    assert result['post2015_boreal_activations'] == []


def test_partial_duplicate_and_malformed_do_not_mean_zero():
    for records, count in (([row()], 2), ([row(), row()], 2), ([row()], True)):
        with pytest.raises(ValueError): summarize(records, count)
    for bad in ({**row(), 'category': {}}, {**row(), 'countries': [42]}):
        with pytest.raises(ValueError): normalize(bad)


def test_blank_or_missing_country_labels_are_preserved_as_unknown():
    for countries in ([{}], [{'short_name': ''}], [' '], [None]):
        result = summarize([{**row(), 'countries': countries}], 1)
        assert len(result['unknown_country_records']) == 1
        assert result['declared_russian_records'] == []
    mixed = {**row(), 'countries': [{'short_name': 'Russia'}, {}]}
    result = summarize([mixed], 1)
    assert len(result['unknown_country_records']) == 1
    assert len(result['declared_russian_records']) == 1
