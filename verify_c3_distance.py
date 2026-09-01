"""
Run this from the project root:  python3 verify_c3_distance.py

Checks C3 (round-trip distance limit) on the real vehicle_routes.csv —
the last constraint not yet directly verified at real scale.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
limit_lookup = dict(zip(tables['vehicles']['vehicle_type'], tables['vehicles']['round_trip_km_limit']))
dist_lookup = {(r.origin, r.destination): r.distance_km for r in tables['distances'].itertuples(index=False)}
chosen_paths = pd.read_csv('data/processed/heuristic_v2_chosen_paths.csv')
path_lookup = {(r.source_node, r.dest_node): r for r in chosen_paths.itertuples(index=False)}

routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])

violations = []
for row in routes.itertuples(index=False):
    limit = limit_lookup.get(row.vehicle_type)
    if limit is None or pd.isna(limit):
        continue  # unbounded type
    if row.mode == 'leg':
        d = dist_lookup.get((row.start_node, row.end_node))
    else:
        prow = path_lookup.get((row.start_node, row.end_node))
        d = prow.total_distance_km if prow is not None else None
    if d is None:
        continue
    round_trip = 2 * d
    if round_trip > limit + 1e-6:
        violations.append({'vehicle_id': row.vehicle_id, 'vehicle_type': row.vehicle_type,
                            'mode': row.mode, 'round_trip_km': round_trip, 'limit_km': limit})

print(f"Total dispatches checked: {len(routes)}")
print(f"C3 (round-trip distance) VIOLATIONS: {len(violations)}")
if violations:
    print(pd.DataFrame(violations).sort_values('round_trip_km', ascending=False).head(15).to_string(index=False))
