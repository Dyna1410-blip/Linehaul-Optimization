"""
Stronger heuristic for path/fleet decisions — an alternative to both the
simple v1 heuristic (optimizer.py) and the MILP (optimizer_milp.py), for
when the MILP's proven gap isn't closing fast enough and a good, honestly
-reported feasible solution is the practical goal.

TWO REAL UPGRADES OVER v1 (optimizer.py):
1. MIXED FLEET — v1 forced a single vehicle type across the whole network.
   v2 picks the most cost-effective vehicle type PER LEG, based on that
   leg's typical daily volume.
2. HOP vs TOUCH per path — v1 never modeled touch cost at all (every
   intermediate stop was priced as a full hop). v2 evaluates both a
   leg-based (hop) and a through-route (touch) estimate per path and picks
   whichever is cheaper — the same trade-off the MILP makes, via a fast
   local rule instead of joint optimization.

KEY MODELING CORRECTION vs v1: Section 2 charges fixed cost PER DISPATCHED
TRIP, not per vehicle owned — there is no cost term for fleet size itself.
That makes fleet size a pure CONSTRAINT (C7: constant daily), not a cost
lever. v1's binary search for the "smallest feasible fleet" was therefore
solving the wrong problem — a smaller fleet saves nothing and only risks
avoidable spillage penalty. v2 instead sizes each vehicle type's fleet at
its PEAK daily need, which is free and minimizes spillage.

APPROACH: iterative (coordinate-descent-style) refinement —
  1. Start with a default vehicle type (largest) for every leg.
  2. Pick each path's mode (leg/through) + vehicle type given current
     per-leg type assignments.
  3. Recompute each leg's realized daily volume from those choices.
  4. Re-pick the best vehicle type per leg given the new volumes.
  5. Repeat a few passes, then do a REAL day-by-day simulation (reusing
     fleet_assignment.py's tested bin-packing) to get the true realized
     cost — not just the rough per-kg estimates used to make selection
     decisions during the iterative passes.

KNOWN LIMITATION: node capacity (C2) is not enforced during construction
(unlike the MILP) — only checked afterward via constraints.py, and
reported, not corrected. A violation found this way would need a manual
fix or another pass, not something this heuristic resolves automatically.
"""

import pandas as pd
import sys
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
import cost_model as cm
import fleet_assignment as fa
from optimizer import select_paths as select_paths_v1


# ---------------------------------------------------------------------------
# Per-leg vehicle type selection
# ---------------------------------------------------------------------------

def estimate_leg_daily_volume(chosen_paths_df: pd.DataFrame, demand_df: pd.DataFrame) -> pd.DataFrame:
    """Average daily physical kg flowing over each leg, given a set of
    chosen paths (used to decide which vehicle type suits each leg)."""
    exploded = fa.explode_orders_to_legs(demand_df, chosen_paths_df)
    if exploded.empty:
        return pd.DataFrame(columns=["leg_from", "leg_to", "avg_daily_phy_kg"])
    n_days = demand_df["date"].nunique()
    totals = exploded.groupby(["leg_from", "leg_to"])["phy_wt_kg"].sum().reset_index()
    totals["avg_daily_phy_kg"] = totals["phy_wt_kg"] / n_days
    return totals[["leg_from", "leg_to", "avg_daily_phy_kg"]]


def pick_vehicle_type_for_leg(daily_volume_kg: float, leg_distance_km: float,
                               vehicle_df: pd.DataFrame) -> str:
    """Among vehicle types whose round-trip limit fits this leg (C3),
    picks whichever gives the lowest effective $/kg for this volume —
    roughly: (vehicles needed to cover the volume) * (round-trip cost) / volume.
    A quick, phy-weight-only proxy for type selection; the later real
    simulation still respects both phy AND vol capacity properly via
    fleet_assignment.pack_orders_into_vehicles.
    """
    import math
    best_type, best_cost_per_kg = None, None
    for v_row in vehicle_df.itertuples(index=False):
        limit = getattr(v_row, "round_trip_km_limit", None)
        if limit is not None and not pd.isna(limit) and 2 * leg_distance_km > limit:
            continue
        if daily_volume_kg <= 0 or v_row.phy_cap_kg <= 0:
            continue
        n_vehicles = max(1, math.ceil(daily_volume_kg / v_row.phy_cap_kg))
        rt_cost = v_row.fixed_cost + v_row.per_km_cost * 2 * leg_distance_km
        cost_per_kg = (n_vehicles * rt_cost) / daily_volume_kg
        if best_cost_per_kg is None or cost_per_kg < best_cost_per_kg:
            best_cost_per_kg = cost_per_kg
            best_type = v_row.vehicle_type
    return best_type


def choose_vehicle_types_per_leg(leg_volumes: pd.DataFrame, dist_df: pd.DataFrame,
                                  vehicle_df: pd.DataFrame, default_type: str) -> dict:
    dist_lookup = {(r.origin, r.destination): r.distance_km for r in dist_df.itertuples(index=False)}
    result = {}
    for row in leg_volumes.itertuples(index=False):
        d = dist_lookup.get((row.leg_from, row.leg_to))
        if d is None:
            result[(row.leg_from, row.leg_to)] = default_type
            continue
        chosen = pick_vehicle_type_for_leg(row.avg_daily_phy_kg, d, vehicle_df)
        result[(row.leg_from, row.leg_to)] = chosen or default_type
    return result


# ---------------------------------------------------------------------------
# Path selection: hop (leg-based) vs touch (through-route), per path
# ---------------------------------------------------------------------------

def estimate_path_mode_and_cost(path_row: pd.Series, weight_kg: float,
                                 leg_vehicle_types: dict, leg_volumes: dict,
                                 vehicle_df: pd.DataFrame,
                                 hop_cost_df: pd.DataFrame, one_way_df: pd.DataFrame,
                                 dist_df: pd.DataFrame, default_type: str,
                                 touch_cost_factor: float = 0.5) -> dict:
    """Returns {"mode": "leg"|"through", "vehicle_type": ..., "cost_per_kg": ...}
    — whichever of leg-based or through-route is cheaper for this path.

    leg_volumes: dict (leg_from, leg_to) -> realized average daily kg on
    that leg (across ALL paths using it, not just this one). This matters:
    leg-based mode's real advantage is SHARING a leg's vehicle with other
    OD flows, amortizing the fixed dispatch cost across everyone using it.
    Without this, leg-mode cost would be computed as if this ONE shipment
    needed a fresh dedicated vehicle per leg — identical in kind to the
    through-route estimate, which always "wins" for multi-leg paths purely
    because it dispatches once instead of twice, regardless of the actual
    hop-vs-touch cost trade-off this function is supposed to be testing.
    (Caught via test: a cheap hop cost still picked "through" until this
    amortization was added — see heuristic_v2 test history.)
    """
    import math
    nodes = path_row["path"].split("|")
    legs = list(zip(nodes[:-1], nodes[1:]))
    intermediate_nodes = nodes[1:-1]

    if weight_kg <= 0:
        return {"mode": "leg", "vehicle_type": default_type, "cost_per_kg": 0.0}

    # --- leg-based (hop) estimate: AMORTIZED over the leg's realized volume ---
    dist_lookup = {(r.origin, r.destination): r.distance_km for r in dist_df.itertuples(index=False)}
    leg_cost_per_kg_total = 0.0
    for (lf, lt) in legs:
        v_type = leg_vehicle_types.get((lf, lt), default_type)
        d = dist_lookup.get((lf, lt), 0.0)
        trip = cm.trip_cost(v_type, d, lf, lt, vehicle_df, one_way_df)
        leg_vol = leg_volumes.get((lf, lt), weight_kg)  # fall back to own weight if unknown
        v_row = cm.lookup_vehicle(v_type, vehicle_df)
        n_vehicles = max(1, math.ceil(leg_vol / v_row["phy_cap_kg"])) if v_row["phy_cap_kg"] else 1
        amortized_cost_per_kg = (n_vehicles * trip["total"]) / leg_vol if leg_vol > 0 else 0.0
        leg_cost_per_kg_total += amortized_cost_per_kg
    hop_per_kg = sum(cm.hop_cost(n, 1.0, hop_cost_df) for n in intermediate_nodes)  # per-kg rate directly
    leg_mode_cost_per_kg = leg_cost_per_kg_total + hop_per_kg

    if not intermediate_nodes:
        # direct path — hop/touch distinction is moot
        return {"mode": "leg", "vehicle_type": leg_vehicle_types.get(legs[0], default_type),
                "cost_per_kg": leg_mode_cost_per_kg}

    # --- through-route (touch) estimate: single dedicated vehicle, full path ---
    best_through_type, best_through_cost = None, None
    for v_row in vehicle_df.itertuples(index=False):
        limit = getattr(v_row, "round_trip_km_limit", None)
        if limit is not None and not pd.isna(limit) and 2 * path_row["total_distance_km"] > limit:
            continue
        trip = cm.trip_cost(v_row.vehicle_type, path_row["total_distance_km"],
                             path_row["source_node"], path_row["dest_node"], vehicle_df, one_way_df)
        n_vehicles = max(1, math.ceil(weight_kg / v_row.phy_cap_kg)) if v_row.phy_cap_kg else 1
        total = n_vehicles * trip["total"]
        if best_through_cost is None or total < best_through_cost:
            best_through_cost = total
            best_through_type = v_row.vehicle_type

    touch_total = sum(cm.touch_cost(n, weight_kg, hop_cost_df, touch_cost_factor) for n in intermediate_nodes)
    through_mode_cost_per_kg = ((best_through_cost or float("inf")) + touch_total) / weight_kg

    if through_mode_cost_per_kg < leg_mode_cost_per_kg:
        return {"mode": "through", "vehicle_type": best_through_type,
                "cost_per_kg": through_mode_cost_per_kg}
    return {"mode": "leg", "vehicle_type": None, "cost_per_kg": leg_mode_cost_per_kg}


def select_paths_v2(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
                     leg_vehicle_types: dict, leg_volumes: dict, vehicle_df: pd.DataFrame,
                     hop_cost_df: pd.DataFrame, one_way_df: pd.DataFrame,
                     dist_df: pd.DataFrame, default_type: str) -> pd.DataFrame:
    avg_weight = demand_df.groupby(["source_node", "dest_node"])["phy_wt_kg"].mean()

    rows = []
    for (source, dest), group in candidate_paths_df.groupby(["source_node", "dest_node"]):
        weight = avg_weight.get((source, dest))
        if weight is None or weight <= 0:
            continue
        best = None
        for path_row in group.itertuples(index=False):
            est = estimate_path_mode_and_cost(
                pd.Series(path_row._asdict()), weight, leg_vehicle_types, leg_volumes,
                vehicle_df, hop_cost_df, one_way_df, dist_df, default_type,
            )
            if best is None or est["cost_per_kg"] < best["cost_per_kg"]:
                best = {**est, "path": path_row.path, "source_node": source, "dest_node": dest,
                        "total_distance_km": path_row.total_distance_km}
        if best is not None:
            rows.append(best)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

def run_iterative_heuristic(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
                             vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                             one_way_df: pd.DataFrame, dist_df: pd.DataFrame,
                             n_iterations: int = 4, verbose: bool = True) -> pd.DataFrame:
    default_type = vehicle_df.loc[vehicle_df["phy_cap_kg"].idxmax(), "vehicle_type"]

    # pass 0: initial paths using default type everywhere
    chosen = select_paths_v1(candidate_paths_df, demand_df, vehicle_df, hop_cost_df,
                              one_way_df, default_type)
    chosen = chosen.rename(columns={"est_cost_per_kg": "cost_per_kg"})
    chosen["mode"] = "leg"
    chosen["vehicle_type"] = None

    for it in range(n_iterations):
        leg_volumes_df = estimate_leg_daily_volume(chosen[["source_node", "dest_node", "path"]], demand_df)
        leg_volumes = {(r.leg_from, r.leg_to): r.avg_daily_phy_kg for r in leg_volumes_df.itertuples(index=False)}
        leg_vehicle_types = choose_vehicle_types_per_leg(leg_volumes_df, dist_df, vehicle_df, default_type)
        new_chosen = select_paths_v2(candidate_paths_df, demand_df, leg_vehicle_types, leg_volumes,
                                      vehicle_df, hop_cost_df, one_way_df, dist_df, default_type)
        if verbose:
            total_est_cost = (new_chosen["cost_per_kg"] *
                               demand_df.groupby(["source_node", "dest_node"])["phy_wt_kg"].mean()
                               .reindex(pd.MultiIndex.from_frame(new_chosen[["source_node", "dest_node"]]))
                               .values).sum()
            print(f"  iteration {it+1}/{n_iterations}: estimated cost/kg-weighted total ~ {total_est_cost:,.0f}")
        chosen = new_chosen

    return chosen


# ---------------------------------------------------------------------------
# Realization: real day-by-day simulation for a TRUE cost figure
# ---------------------------------------------------------------------------

def simulate_month_v2(demand_df: pd.DataFrame, chosen_paths_df: pd.DataFrame,
                       vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                       one_way_df: pd.DataFrame, dist_df: pd.DataFrame) -> dict:
    """Real day-by-day simulation honoring each path's chosen mode
    (leg/through) and vehicle type, using the SAME tested bin-packing
    (fleet_assignment.pack_orders_into_vehicles) as the rest of the
    project — not the rough per-kg estimates used during selection.
    Fleet size per type is set at PEAK daily need (see module docstring
    for why that's the correct, free choice given this cost structure).

    Returns {"allocation": DataFrame, "vehicle_routes": DataFrame,
             "fleet_size": DataFrame, "total_cost": float,
             "cost_breakdown": {...}}
    """
    dist_lookup = {(r.origin, r.destination): r.distance_km for r in dist_df.itertuples(index=False)}
    path_lookup = {
        (row.source_node, row.dest_node): row
        for row in chosen_paths_df.itertuples(index=False)
    }

    allocation_rows, route_rows = [], []
    total_fixed = total_per_km = total_hop = total_touch = 0.0
    vehicle_counter = 0

    for date, day_df in demand_df.groupby("date"):
        leg_groups = {}   # (leg_from, leg_to, vehicle_type) -> list of orders
        through_groups = {}  # path_index -> (vehicle_type, list of orders)

        for order in day_df.itertuples(index=False):
            prow = path_lookup.get((order.source_node, order.dest_node))
            if prow is None:
                continue
            nodes = prow.path.split("|")
            order_id = f"{order.date.date()}_{order.source_node}_{order.dest_node}"
            order_dict = {"order_id": order_id, "phy_wt_kg": order.phy_wt_kg, "vol_wt_kg": order.vol_wt_kg}

            if prow.mode == "through" and len(nodes) > 2:
                key = (prow.source_node, prow.dest_node, prow.path, prow.vehicle_type)
                through_groups.setdefault(key, []).append(order_dict)
            else:
                legs = list(zip(nodes[:-1], nodes[1:]))
                for (lf, lt) in legs:
                    v_type = getattr(prow, "vehicle_type", None)
                    # leg-mode paths don't carry a per-path vehicle_type (that's
                    # decided per-LEG, not per-path) — fall back to the largest type
                    if v_type is None:
                        v_type = vehicle_df.loc[vehicle_df["phy_cap_kg"].idxmax(), "vehicle_type"]
                    leg_groups.setdefault((lf, lt, v_type), []).append(order_dict)

        # --- leg-based (hop) dispatch ---
        for (lf, lt, v_type), orders in leg_groups.items():
            vehicles = fa.pack_orders_into_vehicles(orders, v_type, vehicle_df)
            d = dist_lookup.get((lf, lt), 0.0)
            for veh in vehicles:
                vehicle_counter += 1
                vehicle_id = f"V{date.date()}_{vehicle_counter:06d}"
                trip = cm.trip_cost(v_type, d, lf, lt, vehicle_df, one_way_df)
                total_fixed += trip["fixed"]
                total_per_km += trip["per_km"]
                route_rows.append({"vehicle_id": vehicle_id, "date": date, "vehicle_type": v_type,
                                    "start_node": lf, "end_node": lt, "mode": "leg",
                                    "phy_load_kg": veh["phy_load_kg"], "vol_load_kg": veh["vol_load_kg"]})
                for o in veh["orders"]:
                    allocation_rows.append({"date": date, "order_id": o["order_id"], "vehicle_id": vehicle_id,
                                             "vehicle_type": v_type, "leg_from": lf, "leg_to": lt,
                                             "phy_wt_kg": o["phy_wt_kg"], "vol_wt_kg": o["vol_wt_kg"]})

        # hop cost: charged once per (order, intermediate node) — compute from
        # the ORIGINAL per-order weights on leg-mode paths (not per-leg, to
        # avoid double counting across an order's multiple legs)
        for order in day_df.itertuples(index=False):
            prow = path_lookup.get((order.source_node, order.dest_node))
            if prow is None or prow.mode == "through":
                continue
            nodes = prow.path.split("|")
            for node in nodes[1:-1]:
                total_hop += cm.hop_cost(node, order.phy_wt_kg, hop_cost_df)

        # --- through-route (touch) dispatch ---
        for (o_node, d_node, path_str, v_type), orders in through_groups.items():
            prow_row = path_lookup[(o_node, d_node)]
            vehicles = fa.pack_orders_into_vehicles(orders, v_type, vehicle_df)
            for veh in vehicles:
                vehicle_counter += 1
                vehicle_id = f"V{date.date()}_{vehicle_counter:06d}"
                trip = cm.trip_cost(v_type, prow_row.total_distance_km, o_node, d_node, vehicle_df, one_way_df)
                total_fixed += trip["fixed"]
                total_per_km += trip["per_km"]
                route_rows.append({"vehicle_id": vehicle_id, "date": date, "vehicle_type": v_type,
                                    "start_node": o_node, "end_node": d_node, "mode": "through",
                                    "phy_load_kg": veh["phy_load_kg"], "vol_load_kg": veh["vol_load_kg"]})
                for o in veh["orders"]:
                    allocation_rows.append({"date": date, "order_id": o["order_id"], "vehicle_id": vehicle_id,
                                             "vehicle_type": v_type, "leg_from": o_node, "leg_to": d_node,
                                             "phy_wt_kg": o["phy_wt_kg"], "vol_wt_kg": o["vol_wt_kg"]})

        for order in day_df.itertuples(index=False):
            prow = path_lookup.get((order.source_node, order.dest_node))
            if prow is None or prow.mode != "through":
                continue
            nodes = prow.path.split("|")
            for node in nodes[1:-1]:
                total_touch += cm.touch_cost(node, order.phy_wt_kg, hop_cost_df)

    allocation_df = pd.DataFrame(allocation_rows)
    routes_df = pd.DataFrame(route_rows)

    fleet_size = (routes_df.groupby(["date", "vehicle_type"]).size().reset_index(name="n")
                  .groupby("vehicle_type")["n"].max().reset_index().rename(columns={"n": "fleet_size"})
                  if not routes_df.empty else pd.DataFrame(columns=["vehicle_type", "fleet_size"]))

    total_cost = total_fixed + total_per_km + total_hop + total_touch

    return {
        "allocation": allocation_df,
        "vehicle_routes": routes_df,
        "fleet_size": fleet_size,
        "total_cost": total_cost,
        "cost_breakdown": {
            "fixed": total_fixed, "per_km": total_per_km,
            "hop": total_hop, "touch": total_touch,
        },
    }
