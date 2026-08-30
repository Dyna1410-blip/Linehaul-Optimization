"""
Run this from the project root:  python3 inspect_order.py

Deep-dives ONE specific order flagged as having inconsistent per-leg
totals, to find the actual mechanism causing it.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)

# pick one of the flagged orders from your output
TARGET_DATE = pd.Timestamp('2026-07-01')
TARGET_SOURCE = 'SC_52'
TARGET_DEST = 'SC_123'
TARGET_ORDER_ID = f"{TARGET_DATE.date()}_{TARGET_SOURCE}_{TARGET_DEST}"

print(f"=== Investigating order_id: {TARGET_ORDER_ID} ===\n")

print("--- Raw demand_df rows for this (date, source, dest) ---")
demand_rows = tables['demand'][
    (tables['demand']['date'] == TARGET_DATE) &
    (tables['demand']['source_node'] == TARGET_SOURCE) &
    (tables['demand']['dest_node'] == TARGET_DEST)
]
print(demand_rows.to_string(index=False))
print(f"Number of matching demand rows: {len(demand_rows)} (should be 1 if no duplicates)")

print("\n--- Chosen path for this OD ---")
chosen_paths = pd.read_csv('data/processed/heuristic_v2_chosen_paths.csv')
chosen_row = chosen_paths[
    (chosen_paths['source_node'] == TARGET_SOURCE) &
    (chosen_paths['dest_node'] == TARGET_DEST)
]
print(chosen_row.to_string(index=False))

print("\n--- ALL allocation rows for this order_id ---")
alloc = pd.read_csv('data/processed/heuristic_v2_allocation.csv', parse_dates=['date'])
order_rows = alloc[alloc['order_id'] == TARGET_ORDER_ID]
print(order_rows.to_string(index=False))

print("\n--- Per-leg totals ---")
per_leg = order_rows.groupby(['leg_from', 'leg_to'])['phy_wt_kg'].sum()
print(per_leg)
