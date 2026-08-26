"""
Run this from the project root:  python3 run_optimizer.py
"""
import sys
import time
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl
import optimizer as opt

print("Loading data...")
cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')
print(f"  demand: {tables['demand'].shape}, candidate_paths: {candidate_paths.shape}")

print("Running optimizer (this may take a few minutes)...")
start = time.time()
result = opt.run(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Vehicle type used: {result['vehicle_type']}")
print(f"Fleet size: {result['fleet_size']}")
print(f"Spillage: phy={result['spillage']['phy_spill_pct']:.4f}, "
      f"vol={result['spillage']['vol_spill_pct']:.4f}")
print(f"OD pairs with a chosen path: {len(result['chosen_paths'])}")
print(f"Vehicle-legs deployed: {len(result['vehicle_routes'])}")
print(f"Allocation rows: {len(result['allocation'])}")
