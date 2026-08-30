"""
Run this from the project root:  python3 run_heuristic_v2.py

Runs the stronger v2 heuristic (mixed fleet + hop/touch trade-off) against
your real data, and reports the total cost alongside the MILP's proven
lower bound for an honest comparison — this is the practical alternative
to chasing a tighter branch-and-bound gap on the MILP itself.
"""
import sys
import time
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
import heuristic_v2 as h2

print("Loading data...")
cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')
print(f"  demand: {tables['demand'].shape}, candidate_paths: {candidate_paths.shape}")

print("\nRunning iterative path/fleet-type selection...")
start = time.time()
chosen_paths = h2.run_iterative_heuristic(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'], tables['distances'],
    n_iterations=4, verbose=True,
)
print(f"  path selection done in {time.time()-start:.1f}s")

print("\nRunning full month simulation for realized cost...")
start = time.time()
result = h2.simulate_month_v2(
    tables['demand'], chosen_paths, tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'], tables['distances'],
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Total realized cost: {result['total_cost']:,.2f}")
print(f"Cost breakdown: {result['cost_breakdown']}")
print()
print("Fleet size (peak daily need per type):")
print(result['fleet_size'])
print()
n_through = (chosen_paths['mode'] == 'through').sum()
n_leg = (chosen_paths['mode'] == 'leg').sum()
print(f"Paths using through-route (touch): {n_through}")
print(f"Paths using leg-based (hop): {n_leg}")

# save for comparison against the MILP result and for feeding into reports.py
chosen_paths.to_csv('data/processed/heuristic_v2_chosen_paths.csv', index=False)
result['allocation'].to_csv('data/processed/heuristic_v2_allocation.csv', index=False)
result['vehicle_routes'].to_csv('data/processed/heuristic_v2_vehicle_routes.csv', index=False)
result['fleet_size'].to_csv('data/processed/heuristic_v2_fleet_size.csv', index=False)
print("\nSaved heuristic_v2_*.csv to data/processed/")

print()
print("=" * 60)
print("COMPARE THIS to your MILP's proven lower bound (~504-518M)")
print("and best-found incumbent (~671-683M) to see where this heuristic")
print("solution honestly falls between them.")
print("=" * 60)
