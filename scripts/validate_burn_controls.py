"""SPEC-9: прогон перекрёстной проверки на контрольных пожарах.

Берёт контрольный пожар FIRE-3e785381bdc8 (на котором провалилась проверка в
SPEC-7) и по одному крупнейшему завершённому пожару из трёх широтных поясов,
считает для каждого площадь гари и обогащение по термоточкам.

    python3 scripts/validate_burn_controls.py

Требует FIRMS_MAP_KEY в окружении. Результат пишется в /tmp/spec9_rows.json.
"""
import os, json
from datetime import datetime, timezone
from src import firms
from src.burn import BurnStatus, load_burn_config, run_for_event, aoi_extent_km
from src.confidence import load_thresholds, partition as cpart
from src.events import build_events, load_event_config
from src.persistence import analyse, backfill, load_persistence_config, partition as ppart

CFG="config/krasnoyarsk.toml"
region, sources, _ = firms.load_region(CFG)
dets,_ = backfill(region, sources, os.environ["FIRMS_MAP_KEY"], 60, verbose=False)
kept,_ = cpart(dets, load_thresholds(CFG))
pcfg=load_persistence_config(CFG); kept,_ = ppart(kept, analyse(kept,pcfg), pcfg)
events = build_events(kept, load_event_config(CFG))
cfg = load_burn_config(CFG)
now = datetime.now(timezone.utc)
print(f"событий: {len(events)}", flush=True)

ready = [e for e in events if (now-e.last_seen).days >= 45
         and max(aoi_extent_km(e.padded_bbox(cfg.aoi_pad_m))) <= cfg.max_aoi_km]

# контрольный пожар из SPEC-7 + по одному крупнейшему из трёх широтных поясов
targets = [e for e in ready if e.id == "FIRE-3e785381bdc8"]
for lo, hi in ((51, 60), (60, 66), (66, 78)):
    band = [e for e in ready if lo <= e.centroid[1] < hi and e.id not in {t.id for t in targets}]
    if band:
        targets.append(max(band, key=lambda e: e.total_frp))

rows=[]
for e in targets:
    lon, lat = e.centroid
    print(f"\n--- {e.id}  {lat:.2f},{lon:.2f}  FRP={e.total_frp:.0f}  "
          f"горело {e.first_seen:%m-%d}..{e.last_seen:%m-%d}", flush=True)
    r = run_for_event(e, cfg)
    row = dict(event=e.id, lat=round(lat,3), lon=round(lon,3),
               frp=round(e.total_frp,1), detections=len(e.detections),
               status=r.status.value, doy_gap=r.doy_gap_days,
               enrichment=None if r.enrichment is None else round(r.enrichment,2),
               checked=r.detections_checked,
               masked=round(r.masked_fraction,4), burned_ha=round(r.burned_ha,1),
               before=r.scene_before_date, after=r.scene_after_date,
               tile=r.mgrs_tile, reason=r.reason)
    rows.append(row)
    print(f"    статус={r.status.value} DOY-разрыв={r.doy_gap_days} "
          f"обогащение={row['enrichment']} проверено={r.detections_checked}", flush=True)
    print(f"    сцены {r.scene_before_date} -> {r.scene_after_date} тайл={r.mgrs_tile} "
          f"маска={r.masked_fraction:.1%} гарь={r.burned_ha:.1f} га", flush=True)
    if r.reason: print(f"    причина: {r.reason}", flush=True)

json.dump(rows, open("/tmp/spec9_rows.json","w"), ensure_ascii=False, indent=2)
print("\n=== СВОДКА ===", flush=True)
for r in rows:
    print(f"{r['event']} {r['lat']:6.2f}N  {r['status']:12s} "
          f"DOY={str(r['doy_gap']):>4s} обогащение={str(r['enrichment']):>5s} "
          f"гарь={r['burned_ha']:>9.1f} га", flush=True)
