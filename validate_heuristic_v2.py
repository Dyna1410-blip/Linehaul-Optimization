"""
Run this from the project root:  python3 validate_heuristic_v2.py

Checks the heuristic_v2 output (already saved to data/processed/) against
C2 (node capacity) — the known gap flagged when heuristic_v2.py was built.
Its cost came in BELOW the MILP's proven lower bound, which is only
possible if it's violating a constraint the MILP enforces. This checks
whether C2 is that violation.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
from constraints import check_c2_node_capacity

cfg = dl.load_config('config/config.yaml')
node_df = dl.load_nodes(cfg)
cap_lookup = dict(zip(node_df['node'], node_df['processing_capacity_kg']))

# vehicle_routes.csv: one row per vehicle dispatch on one leg on one day,
# with its actual phy load — this is the real physical load touching each
# node that day (origin, destination, or any intermediate transfer point,
# since a multi-leg order produces multiple route rows, one per leg).
routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])

node_load = {}
for row in routes.itertuples(index=False):
    for node in (row.start_node, row.end_node):
        key = (node, row.date)
        node_load[key] = node_load.get(key, 0.0) + row.phy_load_kg

violations = []
for (node, date), load in node_load.items():
    cap = cap_lookup.get(node)
    if cap is None:
        continue
    result = check_c2_node_capacity(load, cap)
    if not result['pass']:
        violations.append({'node': node, 'date': date, 'load_kg': load,
                            'capacity_kg': cap, 'utilization': result['utilization']})

violations_df = pd.DataFrame(violations)
print(f"Node-day combinations checked: {len(node_load)}")
print(f"C2 (node capacity) VIOLATIONS found: {len(violations_df)}")
if not violations_df.empty:
    print()
    print("Worst violations (highest utilization):")
    print(violations_df.sort_values('utilization', ascending=False).head(15).to_string(index=False))
    print()
    print(f"Max utilization seen: {violations_df['utilization'].max():.2f}x capacity")
