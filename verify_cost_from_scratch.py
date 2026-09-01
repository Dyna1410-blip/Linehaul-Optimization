"""
Run this from the project root:  python3 verify_cost_from_scratch.py

Independently recomputes total cost from the saved vehicle_routes.csv,
using the EXACT SAME cost_model functions the real pipeline uses (not a
hand-rolled formula) — to confirm whether simulate_month_v2's internal
cost accumulation is actually correct, ruling that in or out as the
remaining source of the below-the-bound mystery.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
import cost_model as cm

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)

routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])
chosen_paths = pd.read_csv('data/processed/heuristic_v2_chosen_paths.csv')
allocation = pd.read_csv('data/processed/heuristic_v2_allocation.csv', parse_dates=['date'])

dist_lookup = {(r.origin, r.destination): r.distance_km for r in tables['distances'].itertuples(index=False)}
path_lookup = {(r.source_node, r.dest_node): r for r in chosen_paths.itertuples(index=False)}

total_fixed = total_per_km = 0.0
n_errors = 0

for row in routes.itertuples(index=False):
    if row.mode == 'leg':
        d = dist_lookup.get((row.start_node, row.end_node))
        if d is None:
            n_errors += 1
            continue
        trip = cm.trip_cost(row.vehicle_type, d, row.start_node, row.end_node,
                             tables['vehicles'], tables['one_way_lanes'])
    else:  # through
        prow = path_lookup.get((row.start_node, row.end_node))
        if prow is None:
            n_errors += 1
            continue
        trip = cm.trip_cost(row.vehicle_type, prow.total_distance_km, row.start_node, row.end_node,
                             tables['vehicles'], tables['one_way_lanes'])
    total_fixed += trip['fixed']
    total_per_km += trip['per_km']

print(f"Errors (missing lookups): {n_errors}")
print(f"Independently recomputed fixed cost: {total_fixed:,.2f}")
print(f"Independently recomputed per_km cost: {total_per_km:,.2f}")
print(f"Independently recomputed trip total (fixed+per_km): {total_fixed + total_per_km:,.2f}")
print()
print("Compare against the pipeline's OWN reported numbers from run_heuristic_v2.py:")
print("  fixed: 22,606,522.95")
print("  per_km: 408,893,809.35")
print("  trip total: 431,500,332.30")
