"""
Exact MILP formulation of the line-haul network optimization problem,
solved via Gurobi. Requires a licensed gurobipy install.

v2: models the HOP vs TOUCH trade-off explicitly (v1 assumed every
intermediate stop was a hop). For each candidate path's daily flow, the
solver chooses how much to route:
  - "leg-based" (served_leg): the original mechanism — dispatched leg by
    leg, sharing vehicle capacity with OTHER OD flows on the same leg,
    costed at FULL hop rate at every intermediate node.
  - "through-route" (served_through): a DEDICATED vehicle runs the path's
    entire multi-stop route without unloading, costed at the cheaper
    TOUCH rate (half of hop) at every intermediate node, but that vehicle
    cannot also serve other flows that day (no shared capacity).
The solver picks the cost-minimizing mix per (path, day) — this is the
real hop/touch economic trade-off, not something hard-coded either way.

DECISION VARIABLES
  x[o,d,p]             continuous [0,1] — fixed nominal split ratio of OD
                        (o,d)'s volume routed via candidate path p, every
                        day (C6, by construction).
  served_leg[p,t]       continuous >=0 — leg-based portion of path p's
                        flow on day t.
  served_through[p,t]   continuous >=0 — through-route portion of path
                        p's flow on day t. Only exists for paths with >=1
                        intermediate stop (the distinction is moot for a
                        direct 0-stop path).
  n[leg,t,v]            integer >=0 — vehicles of type v dispatched on a
                        single leg on day t (serves served_leg flow;
                        shared across every path using that leg).
  m[p,t,v]              integer >=0 — vehicles of type v dispatched to run
                        candidate path p's ENTIRE route (round trip) on
                        day t, dedicated to that path's served_through flow.
  FS[v]                 integer >=0 — fleet size for vehicle type v,
                        constant across every day (C7). Shared by both
                        leg-based (n) and through-route (m) dispatches —
                        it's the same physical fleet either way.

OBJECTIVE (minimize)
  sum n[leg,t,v] * leg_trip_cost(leg,v)
  + sum m[p,t,v] * through_trip_cost(p,v)          # full path distance,
                                                     # one-way factor keyed
                                                     # on path's own
                                                     # source/dest
  + sum served_leg[p,t] * hop_cost_per_kg[p]
  + sum served_through[p,t] * touch_cost_per_kg[p]  # = 0.5 * hop rate

CONSTRAINTS
  C1  leg capacity (phy & vol): sum_v n[leg,t,v]*cap[v] >= leg-based flow
      through that leg that day (through-route flow does NOT draw on
      shared leg capacity — it has its own dedicated m[] capacity check)
  C2  node capacity: at a path's OWN origin/destination, ALL flow (leg-
      based + through-route) counts, since loading/unloading always
      happens at true endpoints. At a TRUE INTERMEDIATE stop, only
      served_leg counts — a through-route vehicle passes through without
      being processed there. (This is a stated interpretation of "load
      handled", not an explicit rule in the problem statement — flagged.)
  C3  n[leg,t,v] / m[p,t,v] only exist where the relevant round-trip
      distance (single leg / whole path) fits that vehicle type's limit.
  C6  sum_p x[o,d,p] == 1 for every OD pair.
  C7  sum_legs n[leg,t,v] + sum_paths m[p,t,v] <= FS[v] for every day t —
      same fleet pool serves both dispatch types.
  C9  total spillage (phy, vol separately) <= limit_pct * total demand.
"""

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
import cost_model as cm


def _progress(msg: str, start_time: float):
    """Prints a timestamped progress line — model BUILDING is pure Python
    with no built-in progress signal (unlike Gurobi's own solve log, which
    already streams live). At real scale this build phase can itself take
    a while, so this exists to make it visible rather than looking stuck."""
    elapsed = time.time() - start_time
    print(f"  [{elapsed:6.1f}s] {msg}", flush=True)


def _precompute_path_metadata(candidate_paths_df: pd.DataFrame,
                               hop_cost_df: pd.DataFrame,
                               touch_cost_factor: float = 0.5) -> pd.DataFrame:
    df = candidate_paths_df.copy()
    df["nodes"] = df["path"].str.split("|")
    df["legs"] = df["nodes"].apply(lambda ns: list(zip(ns[:-1], ns[1:])))
    df["intermediate_nodes"] = df["nodes"].apply(lambda ns: ns[1:-1])

    cpk_lookup = dict(zip(hop_cost_df["node"], hop_cost_df["cpk"]))
    df["hop_cost_per_kg"] = df["intermediate_nodes"].apply(
        lambda nodes: sum(cpk_lookup.get(n, 0.0) for n in nodes)
    )
    df["touch_cost_per_kg"] = df["hop_cost_per_kg"] * touch_cost_factor
    df["has_intermediate"] = df["intermediate_nodes"].apply(lambda ns: len(ns) > 0)
    return df


def _precompute_leg_trip_costs(legs: set, vehicle_df: pd.DataFrame,
                                dist_df: pd.DataFrame, one_way_df: pd.DataFrame,
                                start_time: float = None, verbose: bool = False) -> dict:
    dist_lookup = {
        (row.origin, row.destination): row.distance_km
        for row in dist_df.itertuples(index=False)
    }
    costs = {}
    n_legs = len(legs)
    for idx, (leg_from, leg_to) in enumerate(legs):
        if verbose and n_legs > 200 and idx % max(1, n_legs // 10) == 0:
            _progress(f"leg trip costs: {idx}/{n_legs} legs", start_time)
        d = dist_lookup.get((leg_from, leg_to))
        if d is None:
            continue
        for v_row in vehicle_df.itertuples(index=False):
            limit = getattr(v_row, "round_trip_km_limit", None)
            if limit is not None and not pd.isna(limit) and 2 * d > limit:
                continue
            result = cm.trip_cost(v_row.vehicle_type, d, leg_from, leg_to,
                                   vehicle_df, one_way_df)
            costs[(leg_from, leg_to, v_row.vehicle_type)] = result["total"]
    return costs


def _precompute_through_trip_costs(paths: pd.DataFrame, vehicle_df: pd.DataFrame,
                                    one_way_df: pd.DataFrame,
                                    start_time: float = None, verbose: bool = False) -> dict:
    """{(path_index, vehicle_type): trip_cost}, only for (path, vehicle)
    pairs whose FULL round-trip distance fits that vehicle's C3 limit."""
    costs = {}
    n_paths = len(paths)
    for count, (i, prow) in enumerate(paths.iterrows()):
        if verbose and n_paths > 500 and count % max(1, n_paths // 10) == 0:
            _progress(f"through-route trip costs: {count}/{n_paths} paths", start_time)
        if not prow.has_intermediate:
            continue
        for v_row in vehicle_df.itertuples(index=False):
            limit = getattr(v_row, "round_trip_km_limit", None)
            if limit is not None and not pd.isna(limit) and 2 * prow.total_distance_km > limit:
                continue
            result = cm.trip_cost(v_row.vehicle_type, prow.total_distance_km,
                                   prow.source_node, prow.dest_node, vehicle_df, one_way_df)
            costs[(i, v_row.vehicle_type)] = result["total"]
    return costs


def _apply_warm_start(warm_start: dict, paths: pd.DataFrame, x, served_leg, served_through,
                       through_keys, n, m_veh, FS, n_keys, m_keys, vehicle_types,
                       start_time: float = None, verbose: bool = False):
    """Sets Var.Start on every variable found in warm_start's DataFrames,
    matched by semantic key (not raw index). Variables absent from
    warm_start are left untouched (Gurobi's own start-completion heuristic
    fills gaps in a partial MIP start, rather than assuming 0 — safer than
    guessing when a previous solution simply doesn't cover a variable)."""
    n_set = 0
    n_set_by_category = {"x": 0, "served_leg": 0, "served_through": 0, "n": 0, "m": 0, "FS": 0}
    n_attempted_by_category = {"x": 0, "served_leg": 0, "served_through": 0, "n": 0, "m": 0, "FS": 0}

    # x[] from chosen_paths (split_ratio per source/dest/path) — set the
    # chosen path's ratio, then EXPLICITLY zero every other candidate path
    # for that OD (same "unset != 0" issue as n[]/m[] below — only ~552 of
    # ~10,451 total x[] variables are ever nonzero, since each OD picks
    # exactly one path; every other candidate path must be explicit 0).
    if warm_start.get("chosen_paths") is not None and not warm_start["chosen_paths"].empty:
        path_lookup = {
            (row.source_node, row.dest_node, row.path): i
            for i, row in paths.iterrows()
        }
        matched_x_indices = set()
        for row in warm_start["chosen_paths"].itertuples(index=False):
            key = (row.source_node, row.dest_node, row.path)
            n_attempted_by_category["x"] += 1
            if key in path_lookup:
                idx = path_lookup[key]
                x[idx].Start = row.split_ratio
                n_set += 1
                n_set_by_category["x"] += 1
                matched_x_indices.add(idx)
        for idx in paths.index:
            if idx not in matched_x_indices:
                x[idx].Start = 0

    # served_leg[] / served_through[] from served — same explicit-zero
    # treatment: only the CHOSEN path's (path,day) entries are nonzero,
    # every other candidate path's served amount must be explicit 0.
    if warm_start.get("served") is not None and not warm_start["served"].empty:
        path_lookup = {
            (row.source_node, row.dest_node, row.path): i
            for i, row in paths.iterrows()
        }
        matched_served_leg_keys = set()
        matched_served_through_keys = set()
        for row in warm_start["served"].itertuples(index=False):
            key = (row.source_node, row.dest_node, row.path)
            i = path_lookup.get(key)
            n_attempted_by_category["served_leg"] += 1
            if i is None:
                continue
            if (i, row.date) in served_leg:
                served_leg[i, row.date].Start = row.served_leg_kg
                n_set += 1
                n_set_by_category["served_leg"] += 1
                matched_served_leg_keys.add((i, row.date))
            if (i, row.date) in through_keys and (i, row.date) in served_through:
                n_attempted_by_category["served_through"] += 1
                served_through[i, row.date].Start = row.served_through_kg
                n_set += 1
                n_set_by_category["served_through"] += 1
                matched_served_through_keys.add((i, row.date))
        for key in served_leg.keys():
            if key not in matched_served_leg_keys:
                served_leg[key].Start = 0
        for key in through_keys:
            if key in served_through and key not in matched_served_through_keys:
                served_through[key].Start = 0

    # n[] from leg_dispatch — set matched entries, then EXPLICITLY zero every
    # other n[] variable. Leaving unmatched integer variables unset (rather
    # than 0) lets Gurobi's own completion heuristic invent values for them
    # — and with only ~1% of n[]/m[] actually matched, it effectively
    # invented a DIFFERENT dispatch pattern than the heuristic really used
    # (confirmed: loaded objective was 697M, not the heuristic's real 434M,
    # with a warning about "675660 unfixed non-continuous variables").
    # Explicit zeroing forces an exact reconstruction instead.
    if warm_start.get("leg_dispatch") is not None and not warm_start["leg_dispatch"].empty:
        matched_n_keys = set()
        for row in warm_start["leg_dispatch"].itertuples(index=False):
            key = (row.leg_from, row.leg_to, row.date, row.vehicle_type)
            n_attempted_by_category["n"] += 1
            if key in n:
                n[key].Start = row.n_dispatched
                n_set += 1
                n_set_by_category["n"] += 1
                matched_n_keys.add(key)
        for key in n_keys:
            if key not in matched_n_keys:
                n[key].Start = 0

    # m_veh[] from through_dispatch — same explicit-zero treatment
    if warm_start.get("through_dispatch") is not None and not warm_start["through_dispatch"].empty:
        path_lookup = {row.path: i for i, row in paths.iterrows()}
        matched_m_keys = set()
        for row in warm_start["through_dispatch"].itertuples(index=False):
            i = path_lookup.get(row.path)
            n_attempted_by_category["m"] += 1
            if i is None:
                continue
            key = (i, row.date, row.vehicle_type)
            if key in m_veh:
                m_veh[key].Start = row.n_dispatched
                n_set += 1
                n_set_by_category["m"] += 1
                matched_m_keys.add(key)
        for key in m_keys:
            if key not in matched_m_keys:
                m_veh[key].Start = 0

    # FS[] from fleet_size — same treatment: vehicle types the heuristic
    # never used at all (0 fleet needed) must be explicit 0, not left unset.
    if warm_start.get("fleet_size") is not None and not warm_start["fleet_size"].empty:
        matched_fs_types = set()
        for row in warm_start["fleet_size"].itertuples(index=False):
            n_attempted_by_category["FS"] += 1
            if row.vehicle_type in vehicle_types:
                FS[row.vehicle_type].Start = row.fleet_size
                n_set += 1
                n_set_by_category["FS"] += 1
                matched_fs_types.add(row.vehicle_type)
        for v in vehicle_types:
            if v not in matched_fs_types:
                FS[v].Start = 0

    if verbose:
        print(f"  warm start breakdown by category (matched/attempted):")
        for cat in n_set_by_category:
            print(f"    {cat}: {n_set_by_category[cat]}/{n_attempted_by_category[cat]}")
        _progress(f"warm start applied: {n_set} variable starting values set", start_time)


def build_and_solve(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
                     vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                     one_way_df: pd.DataFrame, dist_df: pd.DataFrame,
                     node_df: pd.DataFrame, limit_pct: float = 0.12,
                     time_limit_sec: int = 600, mip_gap: float = 0.02,
                     touch_cost_factor: float = 0.5,
                     spillage_penalty_per_kg: float = None,
                     verbose: bool = True,
                     warm_start: dict = None,
                     root_method: int = None,
                     cuts: int = None,
                     cut_passes: int = None,
                     mip_focus: int = None) -> dict:
    """
    spillage_penalty_per_kg: cost charged in the objective for every kg
    left unserved. WITHOUT this, spillage is "free" apart from the hard
    12% cap (C9) — and since serving more weight costs more (hop/touch
    scale with kg), the solver will happily spill up to the full 12% cap
    purely to save handling cost, even on days with ample spare vehicle
    capacity. That's not a real business behavior; C9's 12% is meant as an
    emergency release valve for genuinely capacity-constrained days, not a
    free cost-saving lever. Defaults to 100x the highest per-kg cost
    observed in the network (hop/touch/trip-cost-per-kg), which is large
    enough to make spillage a last resort while still finite (so the
    model can still legally use up to the 12% cap when truly forced to by
    capacity).

    warm_start: optional dict with the SAME shape as this function's return
    value (chosen_paths, fleet_size, leg_dispatch, through_dispatch, served
    DataFrames) — typically loaded from a previous run's saved CSVs. Sets
    Gurobi's Var.Start on every matching variable, keyed by the same
    semantic identifiers used throughout (path string, leg names, vehicle
    type, date) rather than raw Gurobi/pandas indices — this stays correct
    even if candidate_paths.csv gets re-read in a different row order
    between runs. This does NOT literally resume the previous run's
    branch-and-bound tree (that's gone once the process exits) — it gives
    Gurobi a good starting incumbent so it can spend the new time budget
    improving on it rather than re-discovering one from scratch.
    """
    start_time = time.time()
    if verbose:
        _progress(f"starting build: {len(candidate_paths_df)} candidate paths, "
                   f"{demand_df['date'].nunique()} days", start_time)

    paths = _precompute_path_metadata(candidate_paths_df, hop_cost_df, touch_cost_factor)
    if verbose:
        _progress(f"path metadata precomputed ({len(paths)} paths)", start_time)

    demand_df = demand_df.copy()
    demand_df["vol_density"] = (demand_df["vol_wt_kg"] / demand_df["phy_wt_kg"]).replace(
        [float("inf"), -float("inf")], 0
    ).fillna(0)

    all_legs = set()
    for legs in paths["legs"]:
        all_legs.update(legs)
    if verbose:
        _progress(f"found {len(all_legs)} distinct legs across all paths", start_time)

    leg_trip_costs = _precompute_leg_trip_costs(all_legs, vehicle_df, dist_df, one_way_df,
                                                 start_time, verbose)
    if verbose:
        _progress(f"leg trip costs computed ({len(leg_trip_costs)} leg-vehicle combos)", start_time)

    through_trip_costs = _precompute_through_trip_costs(paths, vehicle_df, one_way_df,
                                                          start_time, verbose)
    if verbose:
        _progress(f"through-route trip costs computed "
                   f"({len(through_trip_costs)} path-vehicle combos)", start_time)

    vehicle_types = vehicle_df["vehicle_type"].tolist()
    days = sorted(demand_df["date"].unique())

    m_model = gp.Model("linehaul_milp_v2")
    if not verbose:
        m_model.setParam("OutputFlag", 0)

    # --- x[o,d,p]: split ratio (C6) ---
    path_idx = list(paths.index)
    x = m_model.addVars(path_idx, lb=0, ub=1, name="x")
    for (o, d), group in paths.groupby(["source_node", "dest_node"]):
        m_model.addConstr(gp.quicksum(x[i] for i in group.index) == 1, name=f"c6_{o}_{d}")
    if verbose:
        _progress(f"x[] split-ratio variables + C6 constraints added ({len(path_idx)} paths)",
                   start_time)

    # --- served_leg / served_through ---
    demand_lookup = {
        (row.source_node, row.dest_node, row.date): (row.phy_wt_kg, row.vol_density)
        for row in demand_df.itertuples(index=False)
    }
    served_keys = []
    for i, prow in paths.iterrows():
        for t in days:
            key = (prow.source_node, prow.dest_node, t)
            if key in demand_lookup:
                served_keys.append((i, t))
    if verbose:
        _progress(f"served_keys built ({len(served_keys)} path-day combinations)", start_time)

    # PERFORMANCE FIX: served_keys/through_keys are checked with `in`
    # repeatedly (thousands to hundreds-of-thousands of times) in the
    # constraint-building loops below. As plain lists, each check scans
    # the whole list — turning what should be O(1) lookups into O(n),
    # and the surrounding loops into effectively O(n^2) or worse. At real
    # scale (300K+ entries) this alone was responsible for build stages
    # taking many minutes each instead of seconds. Sets fix this: `in` on
    # a set is O(1) regardless of size. Iteration behavior (order aside,
    # which nothing here depends on) and gurobipy's addVars are both
    # unaffected by using a set instead of a list.
    served_keys = set(served_keys)

    served_leg = m_model.addVars(served_keys, lb=0, name="served_leg")
    through_keys = {(i, t) for (i, t) in served_keys if paths.loc[i, "has_intermediate"]}
    served_through = m_model.addVars(through_keys, lb=0, name="served_through")

    def through_var(i, t):
        return served_through[i, t] if (i, t) in through_keys else 0.0

    for (i, t) in served_keys:
        prow = paths.loc[i]
        phy_demand, _ = demand_lookup[(prow.source_node, prow.dest_node, t)]
        m_model.addConstr(
            served_leg[i, t] + through_var(i, t) <= x[i] * phy_demand,
            name=f"cap_{i}_{t}",
        )
    if verbose:
        _progress(f"served_leg/served_through variables + capacity constraints added "
                   f"({len(served_keys)} leg + {len(through_keys)} through)", start_time)

    # --- n[leg,t,v] (leg-based) and m[p,t,v] (through-route) ---
    valid_leg_vtypes = {(lf, lt): [] for (lf, lt) in all_legs}
    for (lf, lt, v) in leg_trip_costs:
        valid_leg_vtypes[(lf, lt)].append(v)

    n_keys = [
        (lf, lt, t, v)
        for (lf, lt) in all_legs
        for v in valid_leg_vtypes[(lf, lt)]
        for t in days
    ]
    n = m_model.addVars(n_keys, vtype=GRB.INTEGER, lb=0, name="n")

    valid_path_vtypes = {}
    for (i, v) in through_trip_costs:
        valid_path_vtypes.setdefault(i, []).append(v)

    m_keys = [
        (i, t, v)
        for i in valid_path_vtypes
        for v in valid_path_vtypes[i]
        for t in days
        if (i, t) in through_keys
    ]
    m_veh = m_model.addVars(m_keys, vtype=GRB.INTEGER, lb=0, name="m")

    FS = m_model.addVars(vehicle_types, vtype=GRB.INTEGER, lb=0, name="fleet_size")
    if verbose:
        _progress(f"n[] and m[] dispatch variables added "
                   f"({len(n_keys)} leg-dispatch + {len(m_keys)} through-dispatch)", start_time)

    # --- C7: shared fleet pool across BOTH dispatch types ---
    for v in vehicle_types:
        for t in days:
            legs_for_v = [(lf, lt) for (lf, lt) in all_legs if v in valid_leg_vtypes[(lf, lt)]]
            paths_for_v = [i for i in valid_path_vtypes if v in valid_path_vtypes[i]
                           and (i, t) in through_keys]
            m_model.addConstr(
                gp.quicksum(n[lf, lt, t, v] for (lf, lt) in legs_for_v)
                + gp.quicksum(m_veh[i, t, v] for i in paths_for_v)
                <= FS[v],
                name=f"c7_{v}_{t}",
            )
    if verbose:
        _progress("C7 (fleet-sharing) constraints added", start_time)

    # --- C1: leg capacity — only served_leg flow draws on shared legs ---
    leg_flow_phy = {(lf, lt, t): [] for (lf, lt) in all_legs for t in days}
    leg_flow_vol = {(lf, lt, t): [] for (lf, lt) in all_legs for t in days}
    for i, prow in paths.iterrows():
        for t in days:
            if (i, t) not in served_keys:
                continue
            _, vol_density = demand_lookup[(prow.source_node, prow.dest_node, t)]
            for (lf, lt) in prow.legs:
                leg_flow_phy[(lf, lt, t)].append(served_leg[i, t])
                leg_flow_vol[(lf, lt, t)].append((served_leg[i, t], vol_density))

    vehicle_caps = dict(zip(vehicle_df["vehicle_type"],
                             zip(vehicle_df["phy_cap_kg"], vehicle_df["vol_cap_kg"])))

    for (lf, lt) in all_legs:
        for t in days:
            phy_terms = leg_flow_phy[(lf, lt, t)]
            if not phy_terms:
                continue
            vtypes_here = valid_leg_vtypes[(lf, lt)]
            phy_capacity_expr = gp.quicksum(n[lf, lt, t, v] * vehicle_caps[v][0] for v in vtypes_here)
            m_model.addConstr(phy_capacity_expr >= gp.quicksum(phy_terms), name=f"c1phy_{lf}_{lt}_{t}")

            vol_terms = leg_flow_vol[(lf, lt, t)]
            vol_capacity_expr = gp.quicksum(n[lf, lt, t, v] * vehicle_caps[v][1] for v in vtypes_here)
            vol_flow_expr = gp.quicksum(sv * density for sv, density in vol_terms)
            m_model.addConstr(vol_capacity_expr >= vol_flow_expr, name=f"c1vol_{lf}_{lt}_{t}")
    if verbose:
        _progress("C1 (leg capacity) constraints added", start_time)

    # --- through-route capacity: dedicated, per path (not shared) ---
    for i in valid_path_vtypes:
        for t in days:
            if (i, t) not in through_keys:
                continue
            prow = paths.loc[i]
            _, vol_density = demand_lookup[(prow.source_node, prow.dest_node, t)]
            vtypes_here = valid_path_vtypes[i]
            phy_cap_expr = gp.quicksum(m_veh[i, t, v] * vehicle_caps[v][0] for v in vtypes_here)
            vol_cap_expr = gp.quicksum(m_veh[i, t, v] * vehicle_caps[v][1] for v in vtypes_here)
            m_model.addConstr(phy_cap_expr >= served_through[i, t], name=f"c1through_phy_{i}_{t}")
            m_model.addConstr(vol_cap_expr >= served_through[i, t] * vol_density, name=f"c1through_vol_{i}_{t}")
    if verbose:
        _progress("through-route capacity constraints added", start_time)

    # --- C2: node capacity ---
    # endpoints (source/dest of a path): ALL flow counts (leg + through)
    # true intermediate stops: ONLY served_leg counts (touch = no handling)
    node_cap_lookup = dict(zip(node_df["node"], node_df["processing_capacity_kg"]))
    endpoint_touch = {}
    intermediate_touch = {}
    for i, prow in paths.iterrows():
        endpoint_touch.setdefault(prow.source_node, []).append(i)
        endpoint_touch.setdefault(prow.dest_node, []).append(i)
        for node in prow.intermediate_nodes:
            intermediate_touch.setdefault(node, []).append(i)

    for node, cap in node_cap_lookup.items():
        for t in days:
            terms = []
            for i in endpoint_touch.get(node, []):
                if (i, t) in served_keys:
                    terms.append(served_leg[i, t])
                    if (i, t) in through_keys:
                        terms.append(served_through[i, t])
            for i in intermediate_touch.get(node, []):
                if (i, t) in served_keys:
                    terms.append(served_leg[i, t])  # touch flow excluded here
            if terms:
                m_model.addConstr(gp.quicksum(terms) <= cap, name=f"c2_{node}_{t}")
    if verbose:
        _progress("C2 (node capacity) constraints added", start_time)

    # --- C9: spillage ---
    # spill_phy[o,d,t] is an EXPLICIT variable (not just an implicit gap)
    # so it can be penalized in the objective — see spillage_penalty_per_kg
    # in the docstring for why that penalty is necessary, not optional.
    od_days = sorted(set((o, d, t) for (o, d, t) in demand_lookup.keys()))
    spill_phy = m_model.addVars(od_days, lb=0, name="spill_phy")

    paths_by_od = {}
    for i, prow in paths.iterrows():
        paths_by_od.setdefault((prow.source_node, prow.dest_node), []).append(i)

    for (o, d, t) in od_days:
        phy_demand, _ = demand_lookup[(o, d, t)]
        path_ids = paths_by_od.get((o, d), [])
        served_terms = []
        for i in path_ids:
            if (i, t) in served_keys:
                served_terms.append(served_leg[i, t])
            if (i, t) in through_keys:
                served_terms.append(served_through[i, t])
        m_model.addConstr(
            spill_phy[o, d, t] == phy_demand - gp.quicksum(served_terms),
            name=f"spill_def_{o}_{d}_{t}",
        )

    total_phy = demand_df["phy_wt_kg"].sum()
    total_vol = demand_df["vol_wt_kg"].sum()

    total_served_vol_terms = []
    for (i, t) in served_keys:
        prow = paths.loc[i]
        _, vol_density = demand_lookup[(prow.source_node, prow.dest_node, t)]
        total_served_vol_terms.append(served_leg[i, t] * vol_density)
    for (i, t) in through_keys:
        prow = paths.loc[i]
        _, vol_density = demand_lookup[(prow.source_node, prow.dest_node, t)]
        total_served_vol_terms.append(served_through[i, t] * vol_density)

    m_model.addConstr(
        gp.quicksum(spill_phy[o, d, t] for (o, d, t) in od_days) <= limit_pct * total_phy,
        name="c9_phy",
    )
    m_model.addConstr(
        total_vol - gp.quicksum(total_served_vol_terms) <= limit_pct * total_vol, name="c9_vol"
    )
    if verbose:
        _progress("C9 (spillage) variables + constraints added", start_time)

    # --- objective ---
    leg_cost_term = gp.quicksum(n[lf, lt, t, v] * leg_trip_costs[(lf, lt, v)] for (lf, lt, t, v) in n_keys)
    through_cost_term = gp.quicksum(m_veh[i, t, v] * through_trip_costs[(i, v)] for (i, t, v) in m_keys)
    hop_cost_term = gp.quicksum(served_leg[i, t] * paths.loc[i, "hop_cost_per_kg"] for (i, t) in served_keys)
    touch_cost_term = gp.quicksum(
        served_through[i, t] * paths.loc[i, "touch_cost_per_kg"] for (i, t) in through_keys
    )

    if spillage_penalty_per_kg is None:
        candidate_per_kg_costs = list(paths["hop_cost_per_kg"]) + list(paths["touch_cost_per_kg"])
        if leg_trip_costs:
            candidate_per_kg_costs.append(max(leg_trip_costs.values()) / max(vehicle_df["phy_cap_kg"].min(), 1))
        spillage_penalty_per_kg = 100 * max(candidate_per_kg_costs + [1.0])

    spillage_cost_term = spillage_penalty_per_kg * gp.quicksum(spill_phy[o, d, t] for (o, d, t) in od_days)

    m_model.setObjective(
        leg_cost_term + through_cost_term + hop_cost_term + touch_cost_term + spillage_cost_term,
        GRB.MINIMIZE,
    )
    if verbose:
        m_model.update()  # NumVars/NumConstrs are lazy until update() or optimize()
        _progress(f"objective set — model build complete "
                   f"({m_model.NumVars} variables, {m_model.NumConstrs} constraints). "
                   f"Handing off to Gurobi...", start_time)

    m_model.setParam("TimeLimit", time_limit_sec)
    m_model.setParam("MIPGap", mip_gap)
    if root_method is not None:
        # Neither default simplex nor barrier (Method=2) fixed the "stuck
        # at the root" problem on the real dataset — both got stuck in an
        # expensive cutting-plane generation loop (thousands of MIR cuts)
        # and never reached real branch-and-bound search ("Explored 1
        # nodes" persisted either way). Kept as an option since it's
        # harmless, but cuts/cut_passes/mip_focus below are the more
        # promising levers for that specific symptom.
        m_model.setParam("Method", root_method)
    if cuts is not None:
        # Limits cutting-plane AGGRESSIVENESS (0=off, 1=conservative,
        # 2=aggressive, 3=very aggressive). Lower values mean Gurobi spends
        # less time generating/re-solving-after cuts at the root and moves
        # to branching sooner — trading a weaker bound for actual search
        # progress, which is the direct fix for the "stuck at 1 node"
        # symptom seen on the real dataset.
        m_model.setParam("Cuts", cuts)
    if cut_passes is not None:
        # Hard cap on the NUMBER of cutting-plane rounds at the root,
        # regardless of the Cuts aggressiveness setting above — a second,
        # more direct way to force Gurobi out of the cut-generation loop
        # and into branching after a bounded amount of root refinement.
        m_model.setParam("CutPasses", cut_passes)
    if mip_focus is not None:
        # 1 = prioritize finding good feasible solutions over proving
        # optimality — useful if a good-enough answer within the time
        # budget matters more than a tight proven gap.
        m_model.setParam("MIPFocus", mip_focus)

    if warm_start is not None:
        _apply_warm_start(warm_start, paths, x, served_leg, served_through, through_keys,
                           n, m_veh, FS, n_keys, m_keys, vehicle_types, start_time, verbose)

    m_model.optimize()

    return _extract_solution(m_model, paths, n, m_veh, FS, x, served_leg, served_through,
                              spill_phy, od_days, served_keys, through_keys, n_keys, m_keys,
                              vehicle_types, days, total_phy, total_vol)


def _extract_solution(m_model, paths, n, m_veh, FS, x, served_leg, served_through,
                       spill_phy, od_days, served_keys, through_keys, n_keys, m_keys,
                       vehicle_types, days, total_phy, total_vol) -> dict:
    status_map = {
        GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit_reached",
        GRB.INFEASIBLE: "infeasible", GRB.SUBOPTIMAL: "suboptimal",
    }
    status = status_map.get(m_model.Status, f"status_code_{m_model.Status}")

    if m_model.SolCount == 0:
        empty = pd.DataFrame()
        return {"status": status, "objective": None, "gap": None,
                "runtime_sec": m_model.Runtime, "chosen_paths": empty,
                "fleet_size": empty, "leg_dispatch": empty,
                "through_dispatch": empty, "served": empty, "spillage": None}

    chosen_rows = []
    for i in paths.index:
        val = x[i].X
        if val > 1e-6:
            chosen_rows.append({
                "source_node": paths.loc[i, "source_node"],
                "dest_node": paths.loc[i, "dest_node"],
                "path": paths.loc[i, "path"],
                "split_ratio": val,
            })
    chosen_paths = pd.DataFrame(chosen_rows)

    fleet_size = pd.DataFrame([
        {"vehicle_type": v, "fleet_size": round(FS[v].X)} for v in vehicle_types
    ])

    leg_dispatch = pd.DataFrame([
        {"leg_from": lf, "leg_to": lt, "date": t, "vehicle_type": v, "n_dispatched": round(n[lf, lt, t, v].X)}
        for (lf, lt, t, v) in n_keys if n[lf, lt, t, v].X > 1e-6
    ])

    through_dispatch = pd.DataFrame([
        {"path": paths.loc[i, "path"], "date": t, "vehicle_type": v, "n_dispatched": round(m_veh[i, t, v].X)}
        for (i, t, v) in m_keys if m_veh[i, t, v].X > 1e-6
    ])

    served_rows = []
    for (i, t) in served_keys:
        leg_val = served_leg[i, t].X
        through_val = served_through[i, t].X if (i, t) in through_keys else 0.0
        if leg_val > 1e-6 or through_val > 1e-6:
            served_rows.append({
                "source_node": paths.loc[i, "source_node"],
                "dest_node": paths.loc[i, "dest_node"],
                "path": paths.loc[i, "path"], "date": t,
                "served_leg_kg": leg_val, "served_through_kg": through_val,
            })
    served_df = pd.DataFrame(served_rows)

    total_spilled_phy = sum(spill_phy[key].X for key in od_days)
    spillage = {
        "total_spilled_phy_kg": total_spilled_phy,
        "phy_spill_pct": total_spilled_phy / total_phy if total_phy else 0.0,
    }

    return {
        "status": status,
        "objective": m_model.ObjVal if m_model.SolCount > 0 else None,
        "gap": m_model.MIPGap if hasattr(m_model, "MIPGap") else None,
        "runtime_sec": m_model.Runtime,
        "chosen_paths": chosen_paths,
        "fleet_size": fleet_size,
        "leg_dispatch": leg_dispatch,
        "through_dispatch": through_dispatch,
        "served": served_df,
        "spillage": spillage,
    }