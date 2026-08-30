"""
Run this from the project root:  python3 diagnose_cost_gap.py

The heuristic's 464M total is BELOW the MILP's proven ~504M lower bound —
mathematically impossible for two solutions to the same problem. The 3
node-capacity violations found are too mild to explain a ~$40M gap on
their own. This checks the two next most likely explanations directly:
  1. Weight conservation — is 100% of demand actually being served?
  2. Total dispatch count — how many vehicle trips did each solution
     actually use? (Peak fleet size isn't the same thing — total DISPATCH
     EVENTS across the month is what drives fixed+per-km cost.)
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)

print("=== Weight conservation check ===")
total_demand_phy = tables['demand']['phy_wt_kg'].sum()
print(f"Total demand (phy_wt_kg): {total_demand_phy:,.2f}")

alloc = pd.read_csv('data/processed/heuristic_v2_allocation.csv', parse_dates=['date'])
# each order may appear on multiple rows (multi-leg / split across vehicles);
# for leg-mode paths the SAME order's weight repeats on every leg it crosses,
# so summing raw phy_wt_kg here would double/triple count multi-leg orders.
# Group by (date, order_id) and take the max seen for that order as its
# served amount (every leg of one order carries the same full weight).
per_order = alloc.groupby(['date', 'order_id'])['phy_wt_kg'].max()
total_served_phy = per_order.sum()
print(f"Total served (heuristic, phy_wt_kg): {total_served_phy:,.2f}")
print(f"Difference: {total_demand_phy - total_served_phy:,.2f} "
      f"({(total_demand_phy - total_served_phy) / total_demand_phy * 100:.4f}%)")

print()
print("=== Dispatch count check ===")
routes = pd.read_csv('data/processed/heuristic_v2_vehicle_routes.csv', parse_dates=['date'])
print(f"Total vehicle dispatch events (heuristic): {len(routes)}")
print(f"  of which leg-mode: {(routes['mode']=='leg').sum()}")
print(f"  of which through-mode: {(routes['mode']=='through').sum()}")

print()
print(">>> Compare 'Total vehicle dispatch events' above against the MILP's")
print(">>> leg_dispatch + through_dispatch row counts from your last MILP run")
print(">>> (was 7127 + 310 = 7437 in the last one you shared).")
