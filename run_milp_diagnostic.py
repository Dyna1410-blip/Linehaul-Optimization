"""
DIAGNOSTIC RUN — short (15 min) test of root_method=2 (barrier) to see if
it fixes the "Explored 1 nodes" problem seen in the last two runs (both
the 600s and 16,000s runs spent virtually their entire budget stuck in
the root LP relaxation + cutting-plane phase, never really branching).

Run this from the project root:  python3 run_milp_diagnostic.py

WHAT TO LOOK FOR in the output:
  - "Explored 500 nodes" (or any number well above 1) = barrier method is
    helping, it's actually reaching real branch-and-bound search. Good
    sign — worth committing the full time budget with root_method=2 next.
  - Still "Explored 1 nodes" = the bottleneck isn't the LP method itself.
    Next thing to try: Cuts=1 (less aggressive cutting-plane generation)
    alongside root_method=2, or just accept the current best-found
    solution as the answer and move on to the warm-start bridge.
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

TIME_LIMIT_SEC = 900  # 15 minutes — just enough to see behavior, not a real answer
MIP_GAP = 0.02

print(f"Running SHORT diagnostic (time_limit={TIME_LIMIT_SEC}s, root_method=2/barrier)...")
start = time.time()
result = milp.build_and_solve(
    candidate_paths, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'],
    limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
    time_limit_sec=TIME_LIMIT_SEC, mip_gap=MIP_GAP,
    verbose=True, root_method=2,
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Status: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Gap: {result['gap']}")
print()
print(">>> Scroll up and check the 'Explored N nodes' line in Gurobi's log above.")
print(">>> If N is well above 1, root_method=2 is helping — commit the full")
print(">>> time budget next. If N is still 1, the bottleneck is elsewhere.")
