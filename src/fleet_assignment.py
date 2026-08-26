"""
Given chosen paths (one path per OD pair, per C6) and daily demand, assign
orders to specific vehicles on each leg — producing the granular
(leg, order, vehicle) records needed for report 7.3, and per-vehicle
route/load data needed for report 7.4.

SCOPE / KNOWN LIMITATION (v1):
Each leg of a path gets its own dedicated vehicle round-trip. There is NO
cross-leg vehicle chaining yet — a vehicle carrying freight from A to B
does not continue on to C even if the same path continues A->B->C. This
means every intermediate node on a multi-stop path is treated as a full
HOP (unload + reload), never a cheaper TOUCH (pass-through). Deciding
which legs should actually be chained into one multi-stop vehicle trip —
trading touch-cost savings against routing flexibility — is a genuine
optimization decision, not something to hard-code here. That refinement
belongs in optimizer.py, which can call into this module's building
blocks once it exists.

Similarly, the return leg of every vehicle trip is currently assumed
EMPTY (no backhaul matching against reverse-direction demand). That's
also a real optimization opportunity left for optimizer.py.

Both limitations mean this module currently produces a conservative
UPPER BOUND on hop cost and an upper bound on empty-running distance —
a valid, honest starting point, not the final minimized-cost plan.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import cost_model as cm


# ---------------------------------------------------------------------------
# Path selection placeholder (until optimizer.py exists)
# ---------------------------------------------------------------------------

def choose_shortest_path_per_od(candidate_paths_df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder chooser: picks the shortest-distance candidate path for
    each OD pair. Satisfies C6 trivially (one path per OD, used every day).
    optimizer.py should replace this with a real cost-minimizing choice
    once it exists — this exists purely so fleet_assignment.py is runnable
    and testable on its own before that.
    """
    idx = candidate_paths_df.groupby(["source_node", "dest_node"])[
        "total_distance_km"
    ].idxmin()
    return candidate_paths_df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 1: explode each day's orders onto the legs of their chosen path
# ---------------------------------------------------------------------------

def explode_orders_to_legs(demand_df: pd.DataFrame,
                            chosen_paths_df: pd.DataFrame) -> pd.DataFrame:
    """For each (date, source_node, dest_node) demand row, look up its
    chosen path and produce one row per leg the shipment travels, carrying
    the shipment's full weight on every leg (the whole shipment moves
    through each leg of its own path).

    Returns columns: date, order_id, source_node, dest_node, leg_from,
    leg_to, leg_index, n_legs, phy_wt_kg, vol_wt_kg, path
    """
    path_lookup = {
        (row.source_node, row.dest_node): row.path
        for row in chosen_paths_df.itertuples(index=False)
    }

    rows = []
    for order in demand_df.itertuples(index=False):
        key = (order.source_node, order.dest_node)
        path_str = path_lookup.get(key)
        if path_str is None:
            continue  # no feasible path for this OD -> unserved, handled elsewhere
        nodes = path_str.split("|")
        order_id = f"{order.date.date()}_{order.source_node}_{order.dest_node}"
        for i in range(len(nodes) - 1):
            rows.append({
                "date": order.date,
                "order_id": order_id,
                "source_node": order.source_node,
                "dest_node": order.dest_node,
                "leg_from": nodes[i],
                "leg_to": nodes[i + 1],
                "leg_index": i,
                "n_legs": len(nodes) - 1,
                "phy_wt_kg": order.phy_wt_kg,
                "vol_wt_kg": order.vol_wt_kg,
                "path": path_str,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: bin-pack the orders sharing a (date, leg) onto vehicles
# ---------------------------------------------------------------------------

def pack_orders_into_vehicles(orders: list, vehicle_type: str,
                               vehicle_df: pd.DataFrame) -> list:
    """First-fit-decreasing bin packing with two simultaneous capacity
    dimensions (phy + vol). Orders larger than one vehicle's capacity are
    SPLIT across multiple vehicle instances (proportionally across phy and
    vol, assuming uniform density) — this is what makes load-splitting
    visible in report 7.3, per Section 7.3's explicit requirement.

    orders: list of {"order_id": str, "phy_wt_kg": float, "vol_wt_kg": float}
    Returns: list of vehicle instances, each:
        {"vehicle_type": str, "phy_load_kg": float, "vol_load_kg": float,
         "orders": [{"order_id", "phy_wt_kg", "vol_wt_kg"}]}
    """
    v = cm.lookup_vehicle(vehicle_type, vehicle_df)
    phy_cap, vol_cap = float(v["phy_cap_kg"]), float(v["vol_cap_kg"])

    # mutable working copies, sorted largest-first (FFD heuristic)
    remaining = sorted(
        [dict(o) for o in orders], key=lambda o: o["phy_wt_kg"], reverse=True
    )

    vehicles = []
    guard = 0
    max_iterations = len(remaining) * 10 + 100  # safety against infinite loop

    while any(o["phy_wt_kg"] > 1e-6 for o in remaining):
        guard += 1
        if guard > max_iterations:
            raise RuntimeError(
                "pack_orders_into_vehicles: exceeded iteration guard — "
                "an order likely can't be placed (check for zero/negative "
                "capacity or a data issue)."
            )

        bin_remaining_phy = phy_cap
        bin_remaining_vol = vol_cap
        bin_orders = []
        made_progress = False

        for o in remaining:
            if o["phy_wt_kg"] <= 1e-6:
                continue
            if bin_remaining_phy <= 1e-6 or bin_remaining_vol <= 1e-6:
                continue

            max_phy_fit = min(o["phy_wt_kg"], bin_remaining_phy)
            phy_fraction = max_phy_fit / o["phy_wt_kg"] if o["phy_wt_kg"] > 0 else 1.0

            max_vol_fit = min(o["vol_wt_kg"], bin_remaining_vol) if o["vol_wt_kg"] > 0 else 0.0
            vol_fraction = max_vol_fit / o["vol_wt_kg"] if o["vol_wt_kg"] > 0 else 1.0

            fraction = min(phy_fraction, vol_fraction, 1.0)
            if fraction <= 1e-9:
                continue

            assign_phy = o["phy_wt_kg"] * fraction
            assign_vol = o["vol_wt_kg"] * fraction

            bin_orders.append({
                "order_id": o["order_id"],
                "phy_wt_kg": assign_phy,
                "vol_wt_kg": assign_vol,
            })
            bin_remaining_phy -= assign_phy
            bin_remaining_vol -= assign_vol
            o["phy_wt_kg"] -= assign_phy
            o["vol_wt_kg"] -= assign_vol
            made_progress = True

        if not made_progress:
            raise RuntimeError(
                "pack_orders_into_vehicles: no progress made in a full pass — "
                "an order's weight/volume ratio may not fit this vehicle type "
                "at all (check phy_cap_kg/vol_cap_kg vs order density)."
            )

        vehicles.append({
            "vehicle_type": vehicle_type,
            "phy_load_kg": phy_cap - bin_remaining_phy,
            "vol_load_kg": vol_cap - bin_remaining_vol,
            "orders": bin_orders,
        })

    return vehicles


# ---------------------------------------------------------------------------
# Step 3: orchestrate — explode, aggregate per leg, pack, build allocation
# ---------------------------------------------------------------------------

def assign_fleet(demand_df: pd.DataFrame, chosen_paths_df: pd.DataFrame,
                  vehicle_df: pd.DataFrame, vehicle_type: str = None) -> dict:
    """Top-level orchestrator for one planning horizon.

    vehicle_type: if given, every leg is packed using this single vehicle
    type. If None, defaults to the largest-capacity vehicle type available
    (a reasonable default for consolidation legs) — pass an explicit type
    once optimizer.py is deciding fleet composition per lane.

    Returns:
        {"allocation": DataFrame (leg, order, vehicle — report 7.3),
         "vehicle_routes": DataFrame (one row per vehicle instance — feeds
                            report 7.4, still leg-scoped per the v1 scope
                            note in the module docstring)}
    """
    if vehicle_type is None:
        vehicle_type = vehicle_df.loc[
            vehicle_df["phy_cap_kg"].idxmax(), "vehicle_type"
        ]

    exploded = explode_orders_to_legs(demand_df, chosen_paths_df)
    if exploded.empty:
        return {
            "allocation": pd.DataFrame(),
            "vehicle_routes": pd.DataFrame(),
        }

    allocation_rows = []
    route_rows = []
    vehicle_counter = 0

    for (date, leg_from, leg_to), leg_group in exploded.groupby(
        ["date", "leg_from", "leg_to"]
    ):
        orders = leg_group[["order_id", "phy_wt_kg", "vol_wt_kg"]].to_dict("records")
        vehicles = pack_orders_into_vehicles(orders, vehicle_type, vehicle_df)

        for veh in vehicles:
            vehicle_counter += 1
            vehicle_id = f"V{vehicle_counter:06d}"

            route_rows.append({
                "vehicle_id": vehicle_id,
                "date": date,
                "vehicle_type": veh["vehicle_type"],
                "start_node": leg_from,
                "end_node": leg_to,
                "n_legs": 1,  # v1 scope: one leg per vehicle instance, see module docstring
                "phy_load_kg": veh["phy_load_kg"],
                "vol_load_kg": veh["vol_load_kg"],
                "n_orders_carried": len(veh["orders"]),
                "return_leg": "empty",  # v1 scope: no backhaul matching yet
            })

            for o in veh["orders"]:
                # flag unusually small splits for reviewer attention (7.3)
                v_row = cm.lookup_vehicle(veh["vehicle_type"], vehicle_df)
                phy_share = o["phy_wt_kg"] / v_row["phy_cap_kg"] if v_row["phy_cap_kg"] else 0
                flag = "small_split" if 0 < phy_share < 0.05 else None

                allocation_rows.append({
                    "date": date,
                    "order_id": o["order_id"],
                    "vehicle_id": vehicle_id,
                    "vehicle_type": veh["vehicle_type"],
                    "leg_from": leg_from,
                    "leg_to": leg_to,
                    "phy_wt_kg": o["phy_wt_kg"],
                    "vol_wt_kg": o["vol_wt_kg"],
                    "flag": flag,
                })

    return {
        "allocation": pd.DataFrame(allocation_rows),
        "vehicle_routes": pd.DataFrame(route_rows),
    }