import pytest
from src.comp.cdse_retention_metadata import descriptors, identity, lost_population, query


def product(**values):
    p=dict(Id='00000000-0000-0000-0000-000000000001',
        Name='S2A_MSIL2A_20200809T184919_N0500_R027_T11SKB_20230501T030122.SAFE',
        Online=True, ContentDate=dict(Start='2020-08-09T18:49:19Z'), PublicationDate='2023-05-01T03:01:22Z')
    return {**p, **values}


def test_exact_day_tile_query_and_descriptor():
    sample='hls:subsetted_512x512_HLS.S30.T11SKB.2020222.v1.4_merged.tif'
    assert identity(sample)==('T11SKB','2020-08-09')
    assert '2020-08-10T00:00:00.000Z' in query(sample)['filter']
    r=descriptors([product()],1,'T11SKB','2020-08-09')[0]
    assert r['online_state']=='online' and r['encoding_verified'] is False and r['pixel_admission']=='NOT_TESTED'


def test_offline_unknown_and_placeholder_are_not_admission():
    assert descriptors([product(Online=False)],1,'T11SKB','2020-08-09')[0]['online_state']=='offline'
    assert descriptors([product(Online=None)],1,'T11SKB','2020-08-09')[0]['online_state']=='unknown'
    value=product();value['Name']=value['Name'].replace('N0500','N9999')
    assert descriptors([value],1,'T11SKB','2020-08-09')[0]['processing_baseline_known'] is False


def test_wrong_identity_partial_and_duplicate_rejected():
    for products,count,tile,day in (([product()],2,'T11SKB','2020-08-09'),
        ([product(),product()],2,'T11SKB','2020-08-09'),([product()],1,'T12SVC','2020-08-09'),
        ([product()],1,'T11SKB','2020-08-10')):
        with pytest.raises(ValueError):descriptors(products,count,tile,day)
    assert descriptors([],0,'T11SKB','2020-08-09')==[]


def test_probe_or_reduced_loss_population_cannot_replace_final_manifest():
    manifest=dict(spec_id='SPEC-93',qc_probe=True,ready_for_training=False,
        conservative_event_bound=dict(original_known_events=886,retained_known_events=4,lost_protected=[]))
    with pytest.raises(ValueError):lost_population(manifest)


def test_anonymous_session_does_not_consult_environment_credentials(monkeypatch):
    import requests
    from scripts.audit_cdse_retention_metadata import anonymous_session
    def forbidden_lookup(*args,**kwargs):
        raise AssertionError('environment credentials consulted')
    monkeypatch.setattr(requests.sessions,'get_netrc_auth',forbidden_lookup)
    monkeypatch.setenv('HTTPS_PROXY','https://private.invalid')
    with anonymous_session() as client:
        prepared=client.prepare_request(requests.Request('GET','https://catalogue.dataspace.copernicus.eu/odata/v1/Products'))
        assert 'Authorization' not in prepared.headers
        assert client.merge_environment_settings(prepared.url,{},None,None,None)['proxies']=={}
