"""
Core decision layer: choose one path per OD pair (C6), size a constant
daily fleet (C7), and decide which load spills (C9, <=12% each of phy/vol
weight).

APPROACH — heuristic, not exact MILP:
An exact MILP over 552 OD pairs x ~30 candidate paths x 9 vehicle types x
31 days is not tractable without a commercial solver. This module uses a
three-stage heuristic instead:
  1. select_paths()      — per-OD, pick the candidate path with the lowest
                            estimated cost/kg (cost_model.price_path_estimate).
  2. size_fleet()         — simulate the month UNCONSTRAINED to see daily
                            vehicle need, then search for the SMALLEST
                            constant fleet size (C7) whose resulting
                            spillage stays under the C9 limit. Smallest
                            fleet meeting the constraint = lowest fixed
                            cost, which is the right direction for the
                            "minimize total cost" objective.
  3. simulate_month()     — replay the month with that fixed fleet size;
                            days where demand exceeds capacity spill the
                            excess (biggest legs served first — a stated,
                            documented priority rule, not the only
                            defensible one).

SCOPE LIMITS (consistent with fleet_assignment.py):
- Single vehicle type across the whole network. Choosing vehicle type per
  leg (heterogeneous fleet) is a materially bigger optimization problem —
  flagged here as a known extension, not attempted in this version.
- Inherits fleet_assignment.py's v1 limitations: no cross-leg vehicle
  chaining (every intermediate stop costed as a hop, never a touch), and
  every vehicle's return leg is assumed empty (no backhaul matching).
- Because of the two points above, run() produces a valid, constraint-
  compliant plan and an honest cost figure for THAT plan — not a proof of
  network-wide optimality. Tightening any of these three limits is real
  future work, not something silently assumed away.
"""

import math
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import cost_model as cm
import fleet_assignment as fa


# ---------------------------------------------------------------------------
# Stage 1: path selection (C6 — one path per OD pair for the whole horizon)
# ---------------------------------------------------------------------------

def select_paths(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
                  vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                  one_way_df: pd.DataFrame, vehicle_type: str) -> pd.DataFrame:
    """For each OD pair, pick the candidate path minimizing estimated
    cost/kg, using that OD's AVERAGE DAILY weight (mean phy_wt_kg across
    the days it appears in demand_df) as the representative shipment size
    for cost_model.price_path_estimate(). This is a stated simplifying
    assumption — daily weight actually varies, but the path itself must be
    fixed for the whole month (C6), so a single representative weight is
    needed to make the choice.

    Returns: source_node, dest_node, path, portion_id ("p0" — no volume
    splitting across multiple paths in this version), est_cost_per_kg
    """
    avg_weight = demand_df.groupby(["source_node", "dest_node"])["phy_wt_kg"].mean()

    best_rows = []
    for (source, dest), group in candidate_paths_df.groupby(["source_node", "dest_node"]):
        weight = avg_weight.get((source, dest))
        if weight is None or weight <= 0:
            continue  # OD pair has candidate paths but no actual demand -> skip

        best_cost_per_kg = None
        best_row = None
        for path_row in group.itertuples(index=False):
            estimate = cm.price_path_estimate(
                pd.Series(path_row._asdict()), vehicle_type, weight,
                vehicle_df, hop_cost_df, one_way_df,
            )
            cost_per_kg = estimate["total"] / weight
            if best_cost_per_kg is None or cost_per_kg < best_cost_per_kg:
                best_cost_per_kg = cost_per_kg
                best_row = path_row

        if best_row is not None:
            best_rows.append({
                "source_node": source,
                "dest_node": dest,
                "path": best_row.path,
                "portion_id": "p0",
                "est_cost_per_kg": best_cost_per_kg,
            })

    return pd.DataFrame(best_rows)


# ---------------------------------------------------------------------------
# Stage 2: fleet sizing (C7 constant fleet, C9 spillover <= 12%)
# ---------------------------------------------------------------------------

def compute_unconstrained_daily_usage(demand_df: pd.DataFrame,
                                       chosen_paths_df: pd.DataFrame,
                                       vehicle_df: pd.DataFrame,
                                       vehicle_type: str) -> pd.Series:
    """Runs fleet_assignment with no capacity cap to see how many vehicles
    each day would need if the fleet were unlimited. Returns a Series
    indexed by date -> vehicle count."""
    result = fa.assign_fleet(demand_df, chosen_paths_df, vehicle_df, vehicle_type)
    routes = result["vehicle_routes"]
    if routes.empty:
        return pd.Series(dtype=int)
    return routes.groupby("date").size()


def simulate_day_with_capacity(day_demand_df: pd.DataFrame, chosen_paths_df: pd.DataFrame,
                                vehicle_df: pd.DataFrame, vehicle_type: str,
                                fleet_capacity: int) -> dict:
    """Replays a single day's assignment capped at `fleet_capacity`
    vehicles total (shared across every leg running that day, since it's
    the same physical fleet). Legs are served in order of total weight
    descending (biggest flows first) until the fleet is exhausted; any
    remaining legs' orders are spilled.

    Returns {"allocation": [...], "routes": [...],
             "spilled_phy_kg": float, "spilled_vol_kg": float}
    """
    exploded = fa.explode_orders_to_legs(day_demand_df, chosen_paths_df)
    if exploded.empty:
        return {"allocation": [], "routes": [], "spilled_phy_kg": 0.0, "spilled_vol_kg": 0.0}

    leg_totals = exploded.groupby(["leg_from", "leg_to"])["phy_wt_kg"].sum()
    leg_order = leg_totals.sort_values(ascending=False).index.tolist()

    allocation_rows, route_rows = [], []
    spilled_phy, spilled_vol = 0.0, 0.0
    vehicles_used = 0
    vehicle_counter = 0

    for leg_from, leg_to in leg_order:
        leg_group = exploded[
            (exploded["leg_from"] == leg_from) & (exploded["leg_to"] == leg_to)
        ]
        orders = leg_group[["order_id", "phy_wt_kg", "vol_wt_kg"]].to_dict("records")

        if vehicles_used >= fleet_capacity:
            # no fleet left at all -> this whole leg spills
            spilled_phy += sum(o["phy_wt_kg"] for o in orders)
            spilled_vol += sum(o["vol_wt_kg"] for o in orders)
            continue

        vehicles = fa.pack_orders_into_vehicles(orders, vehicle_type, vehicle_df)
        remaining_capacity = fleet_capacity - vehicles_used

        if len(vehicles) <= remaining_capacity:
            served_vehicles = vehicles
        else:
            served_vehicles = vehicles[:remaining_capacity]
            unserved = vehicles[remaining_capacity:]
            for v in unserved:
                spilled_phy += v["phy_load_kg"]
                spilled_vol += v["vol_load_kg"]

        for veh in served_vehicles:
            vehicle_counter += 1
            vehicles_used += 1
            date_val = day_demand_df["date"].iloc[0]
            vehicle_id = f"V{date_val.date()}_{vehicle_counter:05d}"
            route_rows.append({
                "vehicle_id": vehicle_id, "date": date_val,
                "vehicle_type": veh["vehicle_type"],
                "start_node": leg_from, "end_node": leg_to,
                "phy_load_kg": veh["phy_load_kg"], "vol_load_kg": veh["vol_load_kg"],
                "n_orders_carried": len(veh["orders"]), "return_leg": "empty",
            })
            for o in veh["orders"]:
                allocation_rows.append({
                    "date": date_val, "order_id": o["order_id"],
                    "vehicle_id": vehicle_id, "vehicle_type": veh["vehicle_type"],
                    "leg_from": leg_from, "leg_to": leg_to,
                    "phy_wt_kg": o["phy_wt_kg"], "vol_wt_kg": o["vol_wt_kg"],
                })

    return {
        "allocation": allocation_rows, "routes": route_rows,
        "spilled_phy_kg": spilled_phy, "spilled_vol_kg": spilled_vol,
    }


def simulate_month(demand_df: pd.DataFrame, chosen_paths_df: pd.DataFrame,
                    vehicle_df: pd.DataFrame, vehicle_type: str,
                    fleet_capacity: int) -> dict:
    """Runs simulate_day_with_capacity() across every day in demand_df.
    Returns {"allocation": DataFrame, "vehicle_routes": DataFrame,
             "total_spilled_phy_kg", "total_spilled_vol_kg",
             "total_phy_kg", "total_vol_kg"}"""
    all_alloc, all_routes = [], []
    total_spilled_phy, total_spilled_vol = 0.0, 0.0

    for date, day_df in demand_df.groupby("date"):
        day_result = simulate_day_with_capacity(
            day_df, chosen_paths_df, vehicle_df, vehicle_type, fleet_capacity
        )
        all_alloc.extend(day_result["allocation"])
        all_routes.extend(day_result["routes"])
        total_spilled_phy += day_result["spilled_phy_kg"]
        total_spilled_vol += day_result["spilled_vol_kg"]

    return {
        "allocation": pd.DataFrame(all_alloc),
        "vehicle_routes": pd.DataFrame(all_routes),
        "total_spilled_phy_kg": total_spilled_phy,
        "total_spilled_vol_kg": total_spilled_vol,
        "total_phy_kg": demand_df["phy_wt_kg"].sum(),
        "total_vol_kg": demand_df["vol_wt_kg"].sum(),
    }


def size_fleet(demand_df: pd.DataFrame, chosen_paths_df: pd.DataFrame,
               vehicle_df: pd.DataFrame, vehicle_type: str,
               limit_pct: float = 0.12, max_iterations: int = 30) -> dict:
    """Searches for the SMALLEST constant fleet size (C7) whose resulting
    spillage stays within limit_pct (C9) on BOTH phy and vol weight.
    Starts from the unconstrained peak-day requirement as an upper bound
    and searches downward, since a smaller fleet = lower fixed cost.

    Returns {"fleet_size": int, "spillage": {...}, "search_log": [...]}
    """
    daily_usage = compute_unconstrained_daily_usage(
        demand_df, chosen_paths_df, vehicle_df, vehicle_type
    )
    if daily_usage.empty:
        return {"fleet_size": 0, "spillage": None, "search_log": []}

    peak = int(daily_usage.max())
    low, high = 1, peak
    best_feasible = peak  # peak vehicles always satisfies C9 (0% spillage)
    best_feasible_result = None
    search_log = []

    total_phy = demand_df["phy_wt_kg"].sum()
    total_vol = demand_df["vol_wt_kg"].sum()

    # binary search for the smallest fleet_size where both spill%s <= limit_pct
    iterations = 0
    while low <= high and iterations < max_iterations:
        iterations += 1
        mid = (low + high) // 2
        result = simulate_month(demand_df, chosen_paths_df, vehicle_df, vehicle_type, mid)
        phy_pct = result["total_spilled_phy_kg"] / total_phy if total_phy else 0.0
        vol_pct = result["total_spilled_vol_kg"] / total_vol if total_vol else 0.0
        feasible = phy_pct <= limit_pct + 1e-9 and vol_pct <= limit_pct + 1e-9

        search_log.append({
            "fleet_size": mid, "phy_spill_pct": phy_pct, "vol_spill_pct": vol_pct,
            "feasible": feasible,
        })

        if feasible:
            best_feasible = mid
            best_feasible_result = result
            high = mid - 1  # try smaller
        else:
            low = mid + 1  # need more vehicles

    if best_feasible_result is None:
        best_feasible_result = simulate_month(
            demand_df, chosen_paths_df, vehicle_df, vehicle_type, best_feasible
        )

    return {
        "fleet_size": best_feasible,
        "spillage": {
            "phy_spill_pct": best_feasible_result["total_spilled_phy_kg"] / total_phy if total_phy else 0.0,
            "vol_spill_pct": best_feasible_result["total_spilled_vol_kg"] / total_vol if total_vol else 0.0,
            "total_spilled_phy_kg": best_feasible_result["total_spilled_phy_kg"],
            "total_spilled_vol_kg": best_feasible_result["total_spilled_vol_kg"],
        },
        "search_log": search_log,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run(candidate_paths_df: pd.DataFrame, demand_df: pd.DataFrame,
        vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
        one_way_df: pd.DataFrame, vehicle_type: str = None,
        limit_pct: float = 0.12) -> dict:
    """End-to-end: select paths, size the fleet, produce the final month's
    allocation + vehicle routes under the sized (constant) fleet.

    vehicle_type: if None, defaults to the largest-capacity type (same
    default fleet_assignment.py uses).
    """
    if vehicle_type is None:
        vehicle_type = vehicle_df.loc[vehicle_df["phy_cap_kg"].idxmax(), "vehicle_type"]

    chosen_paths = select_paths(
        candidate_paths_df, demand_df, vehicle_df, hop_cost_df, one_way_df, vehicle_type
    )
    sizing = size_fleet(demand_df, chosen_paths, vehicle_df, vehicle_type, limit_pct)
    final = simulate_month(
        demand_df, chosen_paths, vehicle_df, vehicle_type, sizing["fleet_size"]
    )

    return {
        "vehicle_type": vehicle_type,
        "chosen_paths": chosen_paths,
        "fleet_size": sizing["fleet_size"],
        "spillage": sizing["spillage"],
        "search_log": sizing["search_log"],
        "allocation": final["allocation"],
        "vehicle_routes": final["vehicle_routes"],
    }