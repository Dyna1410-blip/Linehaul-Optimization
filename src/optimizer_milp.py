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

sys.path.insert(0, os.path.dirname(__file__))
import cost_model as cm


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
                                dist_df: pd.DataFrame, one_way_df: pd.DataFrame) -> dict:
    dist_lookup = {
        (row.origin, row.destination): row.distance_km
        for row in dist_df.itertuples(index=False)
    }
    costs = {}
    for (leg_from, leg_to) in legs:
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
                                    one_way_df: pd.DataFrame) -> dict:
    """{(path_index, vehicle_type): trip_cost}, only for (path, vehicle)
    pairs whose FULL round-trip distance fits that vehicle's C3 limit."""
    costs = {}
    for i, prow in paths.iterrows():
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


def build_and_solve(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
                     vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                     one_way_df: pd.DataFrame, dist_df: pd.DataFrame,
                     node_df: pd.DataFrame, limit_pct: float = 0.12,
                     time_limit_sec: int = 600, mip_gap: float = 0.02,
                     touch_cost_factor: float = 0.5,
                     spillage_penalty_per_kg: float = None) -> dict:
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
    """
    paths = _precompute_path_metadata(candidate_paths_df, hop_cost_df, touch_cost_factor)
    demand_df = demand_df.copy()
    demand_df["vol_density"] = (demand_df["vol_wt_kg"] / demand_df["phy_wt_kg"]).replace(
        [float("inf"), -float("inf")], 0
    ).fillna(0)

    all_legs = set()
    for legs in paths["legs"]:
        all_legs.update(legs)

    leg_trip_costs = _precompute_leg_trip_costs(all_legs, vehicle_df, dist_df, one_way_df)
    through_trip_costs = _precompute_through_trip_costs(paths, vehicle_df, one_way_df)
    vehicle_types = vehicle_df["vehicle_type"].tolist()
    days = sorted(demand_df["date"].unique())

    m_model = gp.Model("linehaul_milp_v2")

    # --- x[o,d,p]: split ratio (C6) ---
    path_idx = list(paths.index)
    x = m_model.addVars(path_idx, lb=0, ub=1, name="x")
    for (o, d), group in paths.groupby(["source_node", "dest_node"]):
        m_model.addConstr(gp.quicksum(x[i] for i in group.index) == 1, name=f"c6_{o}_{d}")

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

    served_leg = m_model.addVars(served_keys, lb=0, name="served_leg")
    through_keys = [(i, t) for (i, t) in served_keys if paths.loc[i, "has_intermediate"]]
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

    m_model.setParam("TimeLimit", time_limit_sec)
    m_model.setParam("MIPGap", mip_gap)
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