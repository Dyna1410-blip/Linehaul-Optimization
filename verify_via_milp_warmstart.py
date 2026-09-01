"""
Run this from the project root:  python3 verify_via_milp_warmstart.py

DEFINITIVE TEST: converts the heuristic's actual solution into the MILP's
warm-start format and hands it to Gurobi. Gurobi validates feasibility
itself when loading a MIP start — external, independent proof.

v2: uses heuristic_v2_served.csv (tracked DIRECTLY inside simulate_month_v2
as it computes cost, not reverse-engineered from allocation.csv afterward
via regex string-matching on order_id). The earlier regex approach was a
real, confirmed source of error at real scale — this version has no
reconstruction ambiguity.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
import optimizer_milp as milp

print("Loading data...")
cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')

chosen_paths = pd.read_csv('data/processed/heuristic_v2_chosen_paths.csv')
warm_served = pd.read_csv('data/processed/heuristic_v2_served.csv', parse_dates=['date'])
routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])
fleet_size = pd.read_csv('data/processed/heuristic_v2_fleet_size.csv')

print("Building warm_start dict from heuristic's solution...")

# 1. chosen_paths -> add split_ratio=1.0 (heuristic never splits an OD's flow)
warm_chosen_paths = chosen_paths[['source_node', 'dest_node', 'path']].copy()
warm_chosen_paths['split_ratio'] = 1.0

# 2. served -> loaded DIRECTLY, no reconstruction needed
print(f"  warm_served: {len(warm_served)} rows (loaded directly, not reconstructed)")

# 3. leg_dispatch / through_dispatch directly from vehicle_routes
leg_routes = routes[routes['mode'] == 'leg']
warm_leg_dispatch = leg_routes.groupby(
    ['start_node', 'end_node', 'date', 'vehicle_type']
).size().reset_index(name='n_dispatched').rename(
    columns={'start_node': 'leg_from', 'end_node': 'leg_to'}
)

through_routes = routes[routes['mode'] == 'through']
through_with_path = through_routes.merge(
    chosen_paths[['source_node', 'dest_node', 'path']],
    left_on=['start_node', 'end_node'], right_on=['source_node', 'dest_node'], how='left'
)
warm_through_dispatch = through_with_path.groupby(
    ['path', 'date', 'vehicle_type']
).size().reset_index(name='n_dispatched')

warm_start = {
    'chosen_paths': warm_chosen_paths,
    'served': warm_served,
    'leg_dispatch': warm_leg_dispatch,
    'through_dispatch': warm_through_dispatch,
    'fleet_size': fleet_size,
}

print(f"  warm_chosen_paths: {len(warm_chosen_paths)} rows")
print(f"  warm_leg_dispatch: {len(warm_leg_dispatch)} rows")
print(f"  warm_through_dispatch: {len(warm_through_dispatch)} rows")

print("\nHanding to Gurobi for independent feasibility validation...")
result = milp.build_and_solve(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'],
    limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
    time_limit_sec=120, mip_gap=0.02,
    verbose=True, warm_start=warm_start,
)

print(f"\nStatus: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Gap: {result['gap']}")
print()
print(">>> Look for 'Loaded user MIP start' and its objective value in the")
print(">>> Gurobi log above. If it loaded successfully near 434M, the")
print(">>> heuristic's solution is confirmed feasible and genuinely cheaper.")