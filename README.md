# Middle-Mile Line-Haul Network Optimization

Planning horizon: 31 days (01-07-2026 to 31-07-2026). See `docs/Linehaul_Problem_Statement.pdf`
for the full spec (objective, hard constraints C1-C9, soft constraints S1-S3, deliverables 7.1-7.5).

## Pipeline (in order)

1. **data_loader.py** — load and validate the six raw inputs, normalize column names/types,
   join distance matrix onto OD pairs.
2. **path_enumeration.py** — *(your existing script goes here)* enumerate candidate
   direct + consolidated paths per OD pair, subject to C4 (max 4 intermediate stops),
   C5 (directional/no backtrack), C8 (detour factor cap).
3. **cost_model.py** — per-path, per-vehicle-type cost: fixed + per-km (with one-way factor,
   Section 3) + hop cost + touch cost (Section 2).
4. **constraints.py** — reusable checkers for C1-C9 and S1-S3, used by both the optimizer and
   the report layer (so reports can flag pass/fail without recomputing).
5. **fleet_assignment.py** — assigns orders to vehicles/legs given chosen paths; produces the
   leg-level load-splitting structure needed for report 7.3.
6. **optimizer.py** — the actual path-selection + fleet-size decision (C6 path stability, C7
   constant daily fleet, C9 spillover ≤ 12%). Orchestrates modules above.
7. **reports.py** — builds the five deliverables (7.1-7.5) from the optimizer's output.

## Directory layout

```
data/raw/            # the 6 untouched source files (csv/xlsx) — see below
data/processed/       # cleaned/joined intermediate tables (parquet/csv), gitignored
src/                   # pipeline modules, see above
notebooks/             # exploration / debugging notebooks, not part of the pipeline
config/config.yaml      # paths, tolerances, planning horizon, targets (S1/S2)
config/vehicle_types.yaml  # fleet spec, mirrors vehicle_data_full.csv for quick edits
outputs/reports/       # 5 deliverables land here (one file per report + network summary)
outputs/logs/          # run logs / solver logs
tests/                 # unit tests per module
docs/                  # problem statement + any notes
```

## Expected raw files (drop into `data/raw/`)

| File | Used by |
|---|---|
| `monthly_clustered_loads_combined.csv` | demand — data_loader |
| `node_processing_120.csv` | node capacity — constraints (C2) |
| `one_to_all_distance_matrix_full.csv` | distances — path_enumeration, cost_model |
| `vehicle_data_full.csv` | fleet spec — cost_model, constraints (C1, C3) |
| `B2B Loader CPK Working March-26.csv` | hop/touch cost — cost_model |
| `one_way_lane.xlsx` | one-way pricing — cost_model (Section 3) |

## Status

- [x] Project scaffold
- [ ] Data loader
- [x] Path enumeration (`src/path_enumeration.py` — DFS/branch-and-bound over
      C4/C5/C8; fixed a dest-as-intermediate-node bug on 2026-08-25, see
      module docstring's fix log; covered by `tests/test_path_enumeration.py`)
- [ ] Cost model (stub updated to match path_enumeration's real output schema:
      `path` is a `|`-joined string, not a list — see `src/cost_model.py` header)
- [ ] Constraint checkers (C1-C9, S1-S3)
- [ ] Fleet assignment / optimizer
- [ ] Reports 7.1-7.5

## Running path enumeration

```bash
cd src
python3 path_enumeration.py \
    --distance-matrix ../data/raw/one_to_all_distance_matrix_full.csv \
    --demand ../data/raw/monthly_clustered_loads_combined.csv \
    --output ../data/processed/candidate_paths.csv \
    --top-k 30
```

`--top-k 30` keeps only the 30 shortest candidate paths per OD pair (loose
detour factors like 3.0/4.0/5.0 can otherwise generate thousands of
near-duplicate candidates). Use `--top-k 0` to keep everything.

Known limitation to watch once run on the real 24-node/552-OD-pair data:
worst-case branching is high (up to 23 choices at each of 4 levels before
pruning), so if runtime is a problem on the loose-detour-factor lanes,
memoizing `dist[node, dest]` lookups or vectorizing with numpy is the next
optimization to reach for — not needed yet, flagging for later.
