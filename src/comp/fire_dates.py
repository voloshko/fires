"""SPEC-85: дата начала пожара для окна любого нашего набора: MTBS — ig_date, EMS — eventTime активации, FLOGA — Start date контура."""
import json
from datetime import datetime, timezone
UTC = timezone.utc; _floga = {}


def start_of(w):
    if 'ig_date' in w: return datetime.fromisoformat(w['ig_date']).replace(tzinfo=UTC)
    if 'code' in w: return datetime.fromisoformat(json.load(open(f"external/ems/acts/{w['code']}.json"))['results'][0]['eventTime'][:19]).replace(tzinfo=UTC)
    import geopandas as gpd
    y = int(w['year'])
    if y not in _floga: _floga[y] = gpd.read_file(f'external/floga_ann/fb_{y}.shp').drop(columns='geometry').drop_duplicates('ID').set_index('ID')
    return datetime.fromisoformat(str(_floga[y].loc[int(w['id']), 'Start date'])[:10]).replace(tzinfo=UTC)
