"""
Run this from the project root:  python3 diagnose_cost_gap_v2.py

CORRECTED weight-conservation check — the first version's per_order .max()
undercounted orders that got split across multiple vehicles (each split
fragment is a separate allocation row with only its partial weight).
Fix: sum WITHIN each (date, order_id, leg) first (merges vehicle splits
back into that leg's true total), THEN take one leg's total per order
(every leg of a given order carries the same full weight, by design, so
summing ACROSS legs would double/triple-count a multi-leg order instead).
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
total_demand_phy = tables['demand']['phy_wt_kg'].sum()
print(f"Total demand (phy_wt_kg): {total_demand_phy:,.2f}")

alloc = pd.read_csv('data/processed/heuristic_v2_allocation.csv', parse_dates=['date'])

# Step 1: sum WITHIN each (date, order_id, leg) — correctly merges vehicle
# splits within that one leg back into the leg's true total.
per_leg = alloc.groupby(['date', 'order_id', 'leg_from', 'leg_to'])['phy_wt_kg'].sum().reset_index()

# Step 2: take the FIRST leg's total per order — every leg of a given
# order carries the same full weight (by construction), so any single
# leg's (correctly summed) total already represents the order's true
# served amount; taking more than one would double-count multi-leg orders.
per_order = per_leg.groupby(['date', 'order_id'])['phy_wt_kg'].first()

total_served_phy = per_order.sum()
print(f"Total served (heuristic, CORRECTED): {total_served_phy:,.2f}")
print(f"Difference: {total_demand_phy - total_served_phy:,.2f} "
      f"({(total_demand_phy - total_served_phy) / total_demand_phy * 100:.4f}%)")

# Sanity check: are all legs of the same order actually reporting the SAME
# total (as they should, by design)? If not, that's a real bug to find.
per_order_leg_counts = per_leg.groupby(['date', 'order_id'])['phy_wt_kg'].nunique()
inconsistent = per_order_leg_counts[per_order_leg_counts > 1]
print(f"\nOrders where different legs report DIFFERENT totals "
      f"(should be 0 if correct): {len(inconsistent)}")
if len(inconsistent) > 0:
    print("First few inconsistent orders:")
    print(inconsistent.head(10))
