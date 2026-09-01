"""
Run this from the project root:  python3 diagnose_dispatch_detail.py

Per-km cost (94% of the heuristic's total) is driven by three things:
dispatch COUNT, distance PER dispatch, and vehicle RATE per km. This
breaks all three down so we can see exactly which one differs from the
MILP's own numbers, instead of guessing at the aggregate cost alone.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
dist_lookup = {(r.origin, r.destination): r.distance_km for r in tables['distances'].itertuples(index=False)}
rate_lookup = dict(zip(tables['vehicles']['vehicle_type'], tables['vehicles']['per_km_cost']))
fixed_lookup = dict(zip(tables['vehicles']['vehicle_type'], tables['vehicles']['fixed_cost']))

routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])
print(f"Total dispatch events: {len(routes)}")
print(f"By mode: {routes['mode'].value_counts().to_dict()}")
print(f"By vehicle type: {routes['vehicle_type'].value_counts().to_dict()}")
print()

# reconstruct actual round-trip distance per dispatch
def get_distance(row):
    if row['mode'] == 'through':
        # through mode uses the FULL PATH distance, not the direct leg lookup
        return None  # filled in separately below using chosen_paths
    return dist_lookup.get((row['start_node'], row['end_node']), None)

leg_routes = routes[routes['mode'] == 'leg'].copy()
leg_routes['leg_distance_km'] = leg_routes.apply(get_distance, axis=1)
leg_routes['round_trip_km'] = leg_routes['leg_distance_km'] * 2
leg_routes['rate'] = leg_routes['vehicle_type'].map(rate_lookup)
leg_routes['fixed'] = leg_routes['vehicle_type'].map(fixed_lookup)
leg_routes['per_km_cost'] = leg_routes['round_trip_km'] * leg_routes['rate']

print("=== Leg-mode dispatch summary ===")
print(f"Total leg-mode dispatches: {len(leg_routes)}")
print(f"Total round-trip km run: {leg_routes['round_trip_km'].sum():,.0f}")
print(f"Average round-trip km per dispatch: {leg_routes['round_trip_km'].mean():,.1f}")
print(f"Total per-km cost (leg-mode): {leg_routes['per_km_cost'].sum():,.0f}")
print(f"Total fixed cost (leg-mode): {leg_routes['fixed'].sum():,.0f}")
print()
print("Vehicle type usage breakdown (leg-mode):")
print(leg_routes.groupby('vehicle_type').agg(
    n_dispatches=('vehicle_id', 'count'),
    avg_round_trip_km=('round_trip_km', 'mean'),
    total_per_km_cost=('per_km_cost', 'sum'),
).sort_values('total_per_km_cost', ascending=False))
