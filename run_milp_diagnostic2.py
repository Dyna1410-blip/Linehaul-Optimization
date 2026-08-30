"""
DIAGNOSTIC RUN #2 — root_method=2 (barrier) did NOT fix the stuck-at-1-node
problem (still stuck after 900s, gap 25.7%). This time testing whether
limiting CUTTING-PLANE aggressiveness gets Gurobi into real branching
sooner, since the log showed thousands of cuts (4149 MIR alone) being
generated and re-solved-after at the root — the likely actual bottleneck.

Run this from the project root:  python3 run_milp_diagnostic2.py

WHAT TO LOOK FOR:
  - "Explored 500 nodes" (or any N well above 1) = this fixed it. Commit
    the full time budget with these same settings next.
  - Still "Explored 1 nodes" = cuts aren't the bottleneck either. At that
    point the model may simply be too large for Gurobi to meaningfully
    branch within a practical time budget as currently formulated, and
    accepting the best-found solution (or reducing problem size — fewer
    candidate paths per OD, fewer days) becomes the more practical path
    forward rather than continuing to hunt for a magic parameter.
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

TIME_LIMIT_SEC = 900  # 15 minutes, same as diagnostic #1 for a fair comparison
MIP_GAP = 0.02

print(f"Running SHORT diagnostic #2 (time_limit={TIME_LIMIT_SEC}s, "
      f"cuts=1, cut_passes=5, mip_focus=1)...")
start = time.time()
result = milp.build_and_solve(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'],
    limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
    time_limit_sec=TIME_LIMIT_SEC, mip_gap=MIP_GAP,
    verbose=True, cuts=1, cut_passes=5, mip_focus=1,
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Status: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Gap: {result['gap']}")
print()
print(">>> Check the 'Explored N nodes' line above. N well above 1 = fixed,")
print(">>> commit the full time budget with these settings. Still 1 = try")
print(">>> reducing problem size instead (fewer candidate paths / fewer days).")
