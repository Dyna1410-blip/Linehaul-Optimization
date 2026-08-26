"""
Run this from the project root:  python3 run_milp.py

Requires: pip install gurobipy --break-system-packages
          (and a valid Gurobi license active on this machine)
"""
import sys
import time
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
import optimizer_milp as milp

print("Loading data...")
cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')
print(f"  demand: {tables['demand'].shape}, candidate_paths: {candidate_paths.shape}")

# --- SCALE KNOBS ---
# At full scale (~10,000 candidate paths x 31 days) this is a large MILP,
# now roughly 2x the variable count of the v1 model since it adds
# served_through / m[] variables for every path with >=1 intermediate stop.
# Start small, then dial up:
#   1) Fewer candidate paths per OD: re-run path_enumeration.py with a
#      smaller --top-k (e.g. 10 instead of 30).
#   2) Fewer days: slice demand_df to a subset of dates first, e.g.
#      tables['demand'][tables['demand']['date'] <= '2026-07-07']
#   3) Loosen mip_gap / lower time_limit_sec for a faster, good-enough answer.

TIME_LIMIT_SEC = 600
MIP_GAP = 0.02

print(f"Building and solving MILP (time_limit={TIME_LIMIT_SEC}s, mip_gap={MIP_GAP})...")
start = time.time()
result = milp.build_and_solve(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'],
    limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
    time_limit_sec=TIME_LIMIT_SEC, mip_gap=MIP_GAP,
    verbose=True,  # prints build-stage progress; set False to silence and
                   # only see Gurobi's own solve log
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Status: {result['status']}")
print(f"Objective (total cost): {result['objective']}")
print(f"Optimality gap: {result['gap']}")
print(f"Spillage: {result['spillage']}")
print()
print("Fleet size:")
print(result['fleet_size'])
print()
print(f"OD pairs with a routing plan: {result['chosen_paths']['source_node'].nunique() if not result['chosen_paths'].empty else 0}")
print(f"Leg-based dispatch rows: {len(result['leg_dispatch'])}")
print(f"Through-route dispatch rows: {len(result['through_dispatch'])}")
if not result['served'].empty:
    total_leg = result['served']['served_leg_kg'].sum()
    total_through = result['served']['served_through_kg'].sum()
    print(f"Total served via leg/hop: {total_leg:,.0f} kg")
    print(f"Total served via through-route/touch: {total_through:,.0f} kg")

result['chosen_paths'].to_csv('data/processed/milp_chosen_paths.csv', index=False)
result['fleet_size'].to_csv('data/processed/milp_fleet_size.csv', index=False)
result['leg_dispatch'].to_csv('data/processed/milp_leg_dispatch.csv', index=False)
result['through_dispatch'].to_csv('data/processed/milp_through_dispatch.csv', index=False)
result['served'].to_csv('data/processed/milp_served.csv', index=False)
print("\nSaved milp_chosen_paths.csv, milp_fleet_size.csv, milp_leg_dispatch.csv, "
      "milp_through_dispatch.csv, milp_served.csv to data/processed/")