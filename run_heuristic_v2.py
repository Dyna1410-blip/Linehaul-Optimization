"""
Run this from the project root:  python3 run_heuristic_v2.py

Runs the FIXED v2 heuristic (mixed fleet now actually applied + C2 node
capacity enforced via corrective re-routing) against your real data, and
reports the total cost alongside the MILP's proven lower bound.

Two real bugs were fixed since the last run: (1) mixed-fleet vehicle type
selection was computed but silently discarded before the real simulation
(every leg-mode dispatch fell back to the largest vehicle type), and (2)
node capacity (C2) was never checked or enforced, letting some solutions
route more freight through a node than it could actually handle — which
is why the previous run's cost came in BELOW the MILP's proven floor,
which should be mathematically impossible for a genuinely comparable
solution.
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
chosen_paths, leg_vehicle_types = h2.run_iterative_heuristic(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'], tables['distances'],
    n_iterations=4, verbose=True,
)
print(f"  path selection done in {time.time()-start:.1f}s")

print("\nResolving node capacity (C2) violations via re-routing...")
start = time.time()
resolved = h2.resolve_node_capacity_violations(
    candidate_paths, tables['demand'], chosen_paths, leg_vehicle_types,
    tables['vehicles'], tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'], max_rounds=20, verbose=True,
)
elapsed = time.time() - start
print(f"  resolution done in {elapsed:.1f}s")

result = resolved["result"]
chosen_paths = resolved["chosen_paths"]

if not resolved['remaining_violations'].empty:
    print("\nRerouting couldn't clear all violations (likely the violated node IS")
    print("the shipment's own origin/destination, not an avoidable hop). Applying")
    print("spillage as the final fallback, matching how C9 is meant to be used...")
    spilled = h2.spill_to_fit_remaining_violations(
        tables['demand'], chosen_paths, resolved['leg_vehicle_types'], tables['vehicles'],
        tables['hop_costs'], tables['one_way_lanes'], tables['distances'], tables['nodes'],
        resolved['remaining_violations'], spillage_limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
        verbose=True,
    )
    result = spilled['result']
    final_violations = h2.find_node_capacity_violations(result['vehicle_routes'], tables['nodes'])
    print(f"\nViolations after spillage fix: {len(final_violations)}")
    print(f"Spillage used: {spilled['spill_pct']*100:.4f}% "
          f"(limit: {cfg['hard_constraints']['spillover_limit_pct']*100:.1f}%)")
    print(f"Within C9 limit: {spilled['within_c9_limit']}")
    if not spilled['within_c9_limit']:
        print("!" * 60)
        print("WARNING: required spillage EXCEEDS the 12% C9 cap. This instance")
        print("may be genuinely infeasible under strict constraint compliance —")
        print("worth checking node capacity data, or accepting this as a real finding.")
        print("!" * 60)
    resolved['remaining_violations'] = final_violations

print(f"\nTotal realized cost: {result['total_cost']:,.2f}")
print(f"Cost breakdown: {result['cost_breakdown']}")
print(f"Remaining C2 violations: {len(resolved['remaining_violations'])}")
print(f"Path swaps made to resolve violations: {resolved['n_swaps_made']}")
if not resolved['remaining_violations'].empty:
    print()
    print("!" * 60)
    print("WARNING: violations remain unresolved. This cost figure is NOT")
    print("yet a valid, fully-compliant comparison against the MILP's proven")
    print("lower bound — the solution is still exploiting some capacity it")
    print("shouldn't have access to. See remaining_violations below.")
    print(resolved['remaining_violations'].to_string(index=False))
    print("!" * 60)
print()
print("Fleet size (peak daily need per type):")
print(result['fleet_size'])
print()
n_through = (chosen_paths['mode'] == 'through').sum()
n_leg = (chosen_paths['mode'] == 'leg').sum()
print(f"Paths using through-route (touch): {n_through}")
print(f"Paths using leg-based (hop): {n_leg}")

chosen_paths.to_csv('data/processed/heuristic_v2_chosen_paths.csv', index=False)
result['allocation'].to_csv('data/processed/heuristic_v2_allocation.csv', index=False)
result['vehicle_routes'].to_csv('data/processed/heuristic_v2_vehicle_routes.csv', index=False)
result['fleet_size'].to_csv('data/processed/heuristic_v2_fleet_size.csv', index=False)
print("\nSaved heuristic_v2_*.csv to data/processed/")

print()
print("=" * 60)
print("This cost should now be ABOVE the MILP's proven lower bound (~504M).")
print("If it still isn't, something else needs investigating before trusting it.")
print("=" * 60)