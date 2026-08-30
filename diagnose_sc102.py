"""
Run this from the project root:  python3 diagnose_sc102.py

Identifies exactly which OD pairs still route through the persistently
over-capacity node SC_102, and checks whether ANY of their existing
top-30 candidate paths avoid it — to distinguish "needs more candidate
paths" from "genuinely no route exists avoiding this node."
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')
chosen_paths = pd.read_csv('data/processed/heuristic_v2_chosen_paths.csv')

TARGET_NODE = 'SC_102'

# which currently-chosen ODs still route through SC_102?
still_touching = chosen_paths[chosen_paths['path'].apply(lambda p: TARGET_NODE in p.split('|'))]
print(f"OD pairs currently routed through {TARGET_NODE}: {len(still_touching)}")
print(still_touching[['source_node', 'dest_node', 'path']].to_string(index=False))

print()
print(f"=== For each of these ODs, do ANY of their existing candidate paths avoid {TARGET_NODE}? ===")
for row in still_touching.itertuples(index=False):
    od_candidates = candidate_paths[
        (candidate_paths['source_node'] == row.source_node) &
        (candidate_paths['dest_node'] == row.dest_node)
    ]
    avoiding = od_candidates[~od_candidates['path'].apply(lambda p: TARGET_NODE in p.split('|'))]
    avg_weight = tables['demand'][
        (tables['demand']['source_node'] == row.source_node) &
        (tables['demand']['dest_node'] == row.dest_node)
    ]['phy_wt_kg'].mean()
    print(f"{row.source_node} -> {row.dest_node} (avg {avg_weight:,.0f} kg/day): "
          f"{len(od_candidates)} total candidates, {len(avoiding)} avoid {TARGET_NODE}")
    if len(avoiding) > 0:
        print(f"    example alternative: {avoiding.iloc[0]['path']} "
              f"(detour_ratio={avoiding.iloc[0]['detour_ratio']:.2f})")
