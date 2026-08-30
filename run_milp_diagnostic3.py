"""
DIAGNOSTIC RUN #3 — four independent attempts (default, 27x more time,
barrier method, reduced cuts) all landed in the same place: stuck at 1
node, gap clustering around 24-26%. Strong evidence the bottleneck is
model SIZE, not solver settings. This test directly checks that
hypothesis: truncate candidate_paths.csv down to fewer paths per OD
(smaller model) and see if branching actually starts.

Run this from the project root:  python3 run_milp_diagnostic3.py

WHAT TO LOOK FOR:
  - "Explored N nodes" with N well above 1 = confirms it's a size problem.
    Options from here: permanently work with a smaller top-k, or use
    representative-day clustering instead of all 31 days, or treat this
    smaller/faster model as the practical answer.
  - STILL "Explored 1 nodes" even at this much smaller size = something
    structural in the formulation itself is the problem, not just raw
    scale — a different, deeper conversation about reformulating the
    model would be the next step, not further parameter/size tuning.
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
print(f"  original candidate_paths: {candidate_paths.shape}")

# --- THE ACTUAL TEST: truncate to top-K shortest paths per OD ---
TOP_K = 10  # down from whatever --top-k was used originally (e.g. 30)
candidate_paths_small = (
    candidate_paths
    .sort_values("total_distance_km")
    .groupby(["source_node", "dest_node"], as_index=False)
    .head(TOP_K)
    .reset_index(drop=True)
)
print(f"  truncated to top-{TOP_K}: {candidate_paths_small.shape}")

TIME_LIMIT_SEC = 900  # same 15 min as before, for a fair comparison
MIP_GAP = 0.02

print(f"Running SHORT diagnostic #3 (time_limit={TIME_LIMIT_SEC}s, "
      f"top_k={TOP_K}, default solver settings)...")
start = time.time()
result = milp.build_and_solve(
    candidate_paths_small, tables['demand'], tables['vehicles'],
    tables['hop_costs'], tables['one_way_lanes'],
    tables['distances'], tables['nodes'],
    limit_pct=cfg['hard_constraints']['spillover_limit_pct'],
    time_limit_sec=TIME_LIMIT_SEC, mip_gap=MIP_GAP,
    verbose=True,
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Status: {result['status']}")
print(f"Objective: {result['objective']}")
print(f"Gap: {result['gap']}")
print()
print(">>> Check 'Explored N nodes' above. N well above 1 = size was the")
print(">>> real bottleneck, confirmed. Still 1 = something structural in")
print(">>> the formulation is the issue, not raw size.")
