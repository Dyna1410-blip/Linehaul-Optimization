"""
Run this from the project root:  python3 check_penalty_contribution.py

Checks whether the MILP's artificial spillage penalty (added to stop
Gurobi exploiting free spillage) accounts for the gap between its
proven ~504M lower bound and the heuristic's ~434.5M pure-real-cost
figure. The heuristic tracks zero cost for its 14,422kg of spillage;
the MILP's objective (and therefore its lower bound) includes a large
penalty term for the SAME unavoidable spillage.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import data_loader as dl

cfg = dl.load_config('config/config.yaml')
tables = dl.load_all(cfg)
candidate_paths = pd.read_csv('data/processed/candidate_paths.csv')

# reconstruct hop_cost_per_kg and touch_cost_per_kg for every candidate path,
# exactly as optimizer_milp.py's _precompute_path_metadata does
cpk_lookup = dict(zip(tables['hop_costs']['node'], tables['hop_costs']['cpk']))
touch_cost_factor = 0.5

nodes_per_path = candidate_paths['path'].str.split('|')
intermediate_nodes = nodes_per_path.apply(lambda ns: ns[1:-1])
hop_cost_per_kg = intermediate_nodes.apply(lambda ns: sum(cpk_lookup.get(n, 0.0) for n in ns))
touch_cost_per_kg = hop_cost_per_kg * touch_cost_factor

candidate_per_kg_costs = list(hop_cost_per_kg) + list(touch_cost_per_kg)

# reproduce optimizer_milp.py's exact penalty formula
vehicle_df = tables['vehicles']
# leg_trip_costs aren't easily reconstructed here without full model build,
# but that term (max leg trip cost / min vehicle cap) is usually small
# relative to the hop/touch max — compute the hop/touch-based floor, which
# dominates in practice
spillage_penalty_per_kg = 100 * max(candidate_per_kg_costs + [1.0])

total_spilled_kg = 14422.48  # from the MILP's own reported spillage
penalty_contribution = spillage_penalty_per_kg * total_spilled_kg

print(f"Max hop/touch cost per kg observed in network: {max(candidate_per_kg_costs):.4f}")
print(f"Computed spillage_penalty_per_kg: {spillage_penalty_per_kg:,.2f}")
print(f"Total spilled (MILP's own figure): {total_spilled_kg:,.2f} kg")
print(f"Penalty contribution to MILP's objective/bound: {penalty_contribution:,.2f}")
print()
print(f"MILP's proven lower bound: ~504,000,000")
print(f"Lower bound MINUS penalty contribution: ~{504_000_000 - penalty_contribution:,.0f}")
print(f"Heuristic's real cost (no penalty tracked): 434,477,118")
print()
print("If 'lower bound minus penalty' is now close to or below the heuristic's")
print("cost, the penalty term fully or mostly explains the apparent gap.")
