"""
Cost primitives for Section 2 (objective) + Section 3 (one-way pricing).

SCOPE NOTE — read before using price_path_estimate():
Fixed cost is charged ONCE PER DISPATCHED TRIP, not once per leg (Section 2,
component 1). A single vehicle can run a multi-stop consolidated route and
only pay fixed cost once. That means the *true* cost of moving a shipment
depends on which vehicle(s) actually cover which leg(s) of its path — a
decision made in fleet_assignment.py / optimizer.py, not here.

This module therefore exposes two kinds of function:
  1. Primitives (trip_cost, hop_cost, touch_cost) — the actual building
     blocks used once a real vehicle-to-leg(s) assignment is known. These
     are what fleet_assignment.py and reports.py should call.
  2. price_path_estimate() — a planning-stage estimate ONLY, used by the
     optimizer to rank/screen candidate paths before fleet assignment is
     decided. It assumes a single dedicated vehicle round-trips the whole
     path with every intermediate stop treated as a hop (unload/reload).
     That is a reasonable upper-bound-ish estimate for comparing candidate
     paths, but it is NOT the final costed number — real consolidation
     with other orders will usually make the realized cost lower (shared
     fixed cost) and may turn some hops into touches (cheaper).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def lookup_vehicle(vehicle_type: str, vehicle_df: pd.DataFrame) -> pd.Series:
    match = vehicle_df.loc[vehicle_df["vehicle_type"] == vehicle_type]
    if match.empty:
        raise ValueError(
            f"Unknown vehicle_type '{vehicle_type}'. "
            f"Known types: {vehicle_df['vehicle_type'].unique().tolist()}"
        )
    return match.iloc[0]


def lookup_one_way_factor(origin: str, dest: str, one_way_df: pd.DataFrame):
    """Returns the pct (0-1) of round-trip cost charged for a one-way trip
    on this lane, or None if the lane isn't in the discount table (meaning:
    charge the full round-trip cost, per Section 3)."""
    match = one_way_df.loc[
        (one_way_df["origin"] == origin) & (one_way_df["destination"] == dest)
    ]
    if match.empty:
        return None
    return float(match.iloc[0]["pct"])


def round_trip_cost(vehicle_type: str, one_way_distance_km: float,
                     vehicle_df: pd.DataFrame) -> float:
    """Full round-trip cost (fixed + per-km * round-trip distance) for a
    vehicle covering `one_way_distance_km` out and the same distance back."""
    v = lookup_vehicle(vehicle_type, vehicle_df)
    round_trip_km = 2 * one_way_distance_km
    return float(v["fixed_cost"] + v["per_km_cost"] * round_trip_km)


def trip_cost(vehicle_type: str, one_way_distance_km: float, origin: str, dest: str,
              vehicle_df: pd.DataFrame, one_way_df: pd.DataFrame) -> dict:
    """Cost of dispatching one trip on lane origin->dest, per Section 3:
    always the round-trip cost, UNLESS the lane is listed in one_way_lane
    (one_way_df), in which case it's `pct` of that round-trip cost.

    Returns {"fixed": ..., "per_km": ..., "total": ..., "one_way_pct_applied": ...}
    """
    v = lookup_vehicle(vehicle_type, vehicle_df)
    round_trip_km = 2 * one_way_distance_km
    fixed = float(v["fixed_cost"])
    per_km = float(v["per_km_cost"] * round_trip_km)
    rt_total = fixed + per_km

    pct = lookup_one_way_factor(origin, dest, one_way_df)
    if pct is not None:
        fixed *= pct
        per_km *= pct

    return {
        "fixed": fixed,
        "per_km": per_km,
        "total": fixed + per_km,
        "one_way_pct_applied": pct,
    }


def node_cpk(node: str, hop_cost_df: pd.DataFrame) -> float:
    match = hop_cost_df.loc[hop_cost_df["node"] == node]
    if match.empty:
        raise ValueError(
            f"No hop cost (CPK) found for node '{node}'. "
            f"Known nodes: {hop_cost_df['node'].unique().tolist()}"
        )
    return float(match.iloc[0]["cpk"])


def hop_cost(node: str, weight_kg: float, hop_cost_df: pd.DataFrame) -> float:
    """Full hop cost: freight is unloaded and re-handled at `node`."""
    return node_cpk(node, hop_cost_df) * weight_kg


def touch_cost(node: str, weight_kg: float, hop_cost_df: pd.DataFrame,
               touch_cost_factor: float = 0.5) -> float:
    """Touch cost: vehicle stops at `node` but this shipment's freight
    stays loaded (not unloaded) — charged at touch_cost_factor x hop rate."""
    return touch_cost_factor * node_cpk(node, hop_cost_df) * weight_kg


# ---------------------------------------------------------------------------
# Planning-stage estimate (see module docstring — NOT final costing)
# ---------------------------------------------------------------------------

def price_path_estimate(path_row: pd.Series, vehicle_type: str, weight_kg: float,
                         vehicle_df: pd.DataFrame, hop_cost_df: pd.DataFrame,
                         one_way_df: pd.DataFrame,
                         touch_cost_factor: float = 0.5) -> dict:
    """Estimate the cost of moving `weight_kg` along a single candidate path
    (a row from candidate_paths.csv) as if ONE dedicated vehicle round-trips
    the whole path, treating every intermediate stop as a hop.

    path_row columns expected (from path_enumeration.py's output):
        source_node, dest_node, path ("A|B|D" pipe-separated string),
        total_distance_km, ...

    Returns:
        {"fixed": ..., "per_km": ..., "hop": ..., "touch": ..., "total": ...,
         "intermediate_nodes": [...]}
    """
    origin = path_row["source_node"]
    dest = path_row["dest_node"]
    nodes = path_row["path"].split("|")
    intermediate_nodes = nodes[1:-1]

    trip = trip_cost(vehicle_type, float(path_row["total_distance_km"]),
                      origin, dest, vehicle_df, one_way_df)

    hop_total = sum(hop_cost(n, weight_kg, hop_cost_df) for n in intermediate_nodes)
    # touch_total is 0 here by construction (single dedicated vehicle -> every
    # stop is a hop, never a touch); kept as a field so the return shape
    # matches what fleet_assignment.py will produce once touches are real.
    touch_total = 0.0

    return {
        "fixed": trip["fixed"],
        "per_km": trip["per_km"],
        "hop": hop_total,
        "touch": touch_total,
        "total": trip["total"] + hop_total + touch_total,
        "intermediate_nodes": intermediate_nodes,
    }
