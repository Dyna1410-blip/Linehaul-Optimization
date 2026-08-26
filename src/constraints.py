"""
Reusable pass/fail checkers for hard constraints C1-C9 and soft targets
S1-S3 (Sections 5-6 of the problem statement). Used by both optimizer.py
(to prune/select) and reports.py (to annotate deliverables without
recomputing).

Each checker takes plain values or small DataFrames (not the whole
pipeline's state) so it can be unit tested in isolation and reused from
different call sites (a single candidate path during path selection, a
finished vehicle route during reporting, etc).

C5 note: path_enumeration.py already enforces directional routing (no
backtracking) while generating candidate paths, so anything from
candidate_paths.csv should already satisfy C5 by construction. The C5
checker here is still an INDEPENDENT recomputation from the raw node
sequence + distance matrix (not "trust because it came from the
enumerator") — this matters once paths get touched by anything other than
path_enumeration.py (manual overrides, S3 rewiring, etc.), where an
independent check will actually catch a violation instead of rubber-
stamping it.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# C1 - Vehicle capacity
# ---------------------------------------------------------------------------

def check_c1_vehicle_capacity(phy_wt_kg: float, vol_wt_kg: float,
                               phy_cap_kg: float, vol_cap_kg: float) -> dict:
    """Both physical and volumetric weight must independently fit."""
    phy_ok = phy_wt_kg <= phy_cap_kg + 1e-9
    vol_ok = vol_wt_kg <= vol_cap_kg + 1e-9
    return {
        "pass": phy_ok and vol_ok,
        "phy_pass": phy_ok,
        "vol_pass": vol_ok,
        "phy_utilization": phy_wt_kg / phy_cap_kg if phy_cap_kg else None,
        "vol_utilization": vol_wt_kg / vol_cap_kg if vol_cap_kg else None,
    }


# ---------------------------------------------------------------------------
# C2 - Node processing capacity
# ---------------------------------------------------------------------------

def check_c2_node_capacity(node_load_kg: float, processing_capacity_kg: float) -> dict:
    """node_load_kg = incoming + outgoing + pass-through load at the node
    on a single day (the spec: processing_capacity_kgs is already the
    combined in-plus-out limit)."""
    ok = node_load_kg <= processing_capacity_kg + 1e-9
    return {
        "pass": ok,
        "utilization": node_load_kg / processing_capacity_kg if processing_capacity_kg else None,
    }


# ---------------------------------------------------------------------------
# C3 - Maximum round-trip distance
# ---------------------------------------------------------------------------

def check_c3_round_trip_distance(round_trip_km: float, limit_km) -> dict:
    """limit_km may be None/NaN, meaning unbounded (the two 32 FT types)."""
    if limit_km is None or pd.isna(limit_km):
        return {"pass": True, "limit_km": None, "unbounded": True}
    ok = round_trip_km <= limit_km + 1e-9
    return {"pass": ok, "limit_km": limit_km, "unbounded": False}


# ---------------------------------------------------------------------------
# C4 - Maximum intermediate stops
# ---------------------------------------------------------------------------

def check_c4_max_stops(n_intermediate_stops: int, max_stops: int = 4) -> dict:
    """max_stops defaults to the spec's flat cap of 4, but the real
    vehicle_data_full.csv carries a max_intermediate_stops column per
    vehicle type — pass that value in explicitly when checking a specific
    vehicle's leg, rather than relying on the default."""
    ok = n_intermediate_stops <= max_stops
    return {"pass": ok, "max_stops": max_stops}


# ---------------------------------------------------------------------------
# C5 - Directional routing (independent recheck, see module docstring)
# ---------------------------------------------------------------------------

def check_c5_directional(stop_sequence: list, dist_df: pd.DataFrame) -> dict:
    """stop_sequence: ordered list of node names, e.g. ["A", "B", "D"].
    Recomputes forward progress from the distance matrix directly — does
    NOT assume the sequence came from path_enumeration.py.

    A route is valid if, for every consecutive pair of stops (current,
    next), the remaining distance from `next` to the final destination is
    <= the remaining distance from `current` to the final destination.
    """
    if len(stop_sequence) < 2:
        return {"pass": True, "violations": []}

    dest = stop_sequence[-1]
    dist_lookup = {
        (row.origin, row.destination): row.distance_km
        for row in dist_df.itertuples(index=False)
    }

    violations = []
    for i in range(len(stop_sequence) - 1):
        current = stop_sequence[i]
        nxt = stop_sequence[i + 1]
        dist_curr_to_dest = dist_lookup.get((current, dest))
        dist_next_to_dest = dist_lookup.get((nxt, dest))
        if dist_curr_to_dest is None or dist_next_to_dest is None:
            violations.append({
                "from": current, "to": nxt,
                "reason": "missing distance entry for forward-progress check",
            })
            continue
        if dist_next_to_dest > dist_curr_to_dest + 1e-9:
            violations.append({
                "from": current, "to": nxt,
                "reason": f"remaining distance increased "
                          f"({dist_curr_to_dest:.2f} -> {dist_next_to_dest:.2f})",
            })

    return {"pass": len(violations) == 0, "violations": violations}


# ---------------------------------------------------------------------------
# C6 - Path stability over the month
# ---------------------------------------------------------------------------

def check_c6_path_stability(assignment_df: pd.DataFrame) -> dict:
    """assignment_df: one row per (date, source_node, dest_node, portion_id,
    path) — the path assigned to a given split portion of an OD pair on a
    given day. A portion_id must map to exactly ONE path across every date
    it appears (the spec: "each split portion must follow the same path
    every day of the month").

    Returns which portion_ids violate this, if any.
    """
    required_cols = {"date", "source_node", "dest_node", "portion_id", "path"}
    missing = required_cols - set(assignment_df.columns)
    if missing:
        raise ValueError(f"assignment_df missing required columns: {missing}")

    grouped = assignment_df.groupby(
        ["source_node", "dest_node", "portion_id"]
    )["path"].nunique()
    violating = grouped[grouped > 1]

    return {
        "pass": violating.empty,
        "n_portions_checked": len(grouped),
        "n_violations": len(violating),
        "violations": violating.index.tolist(),
    }


# ---------------------------------------------------------------------------
# C7 - Constant daily fleet
# ---------------------------------------------------------------------------

def check_c7_constant_daily_fleet(fleet_by_day_df: pd.DataFrame) -> dict:
    """fleet_by_day_df: columns [date, vehicle_type, n_deployed]. The count
    per vehicle_type must be identical on every day of the horizon."""
    required_cols = {"date", "vehicle_type", "n_deployed"}
    missing = required_cols - set(fleet_by_day_df.columns)
    if missing:
        raise ValueError(f"fleet_by_day_df missing required columns: {missing}")

    grouped = fleet_by_day_df.groupby("vehicle_type")["n_deployed"].nunique()
    violating = grouped[grouped > 1]

    return {
        "pass": violating.empty,
        "violating_vehicle_types": violating.index.tolist(),
    }


# ---------------------------------------------------------------------------
# C8 - Detour factor (independent recheck)
# ---------------------------------------------------------------------------

def check_c8_detour_factor(routed_km: float, direct_km: float, detour_factor: float) -> dict:
    if direct_km is None or direct_km <= 0:
        return {"pass": False, "reason": "no valid direct distance to compute detour ratio"}
    ratio = routed_km / direct_km
    ok = routed_km <= detour_factor * direct_km + 1e-9
    return {"pass": ok, "detour_ratio": ratio, "detour_factor_limit": detour_factor}


# ---------------------------------------------------------------------------
# C9 - Unassigned spillover within 12%
# ---------------------------------------------------------------------------

def check_c9_spillover(spilled_phy_kg: float, spilled_vol_kg: float,
                        total_phy_kg: float, total_vol_kg: float,
                        limit_pct: float = 0.12) -> dict:
    """Applies separately to physical and volumetric weight — a plan
    passing on one measure but breaching the other is still non-compliant."""
    phy_pct = spilled_phy_kg / total_phy_kg if total_phy_kg else 0.0
    vol_pct = spilled_vol_kg / total_vol_kg if total_vol_kg else 0.0
    phy_ok = phy_pct <= limit_pct + 1e-9
    vol_ok = vol_pct <= limit_pct + 1e-9
    return {
        "pass": phy_ok and vol_ok,
        "phy_pass": phy_ok,
        "vol_pass": vol_ok,
        "phy_spill_pct": phy_pct,
        "vol_spill_pct": vol_pct,
        "limit_pct": limit_pct,
    }


# ---------------------------------------------------------------------------
# S1 - Weighted hops per kg within target
# ---------------------------------------------------------------------------

def check_s1_weighted_hops(orders_df: pd.DataFrame, target: float = 0.2) -> dict:
    """orders_df: columns [weight_kg, n_hops] — one row per served order.
    weighted_hops_per_kg = sum(weight_kg * n_hops) / sum(weight_kg)."""
    required_cols = {"weight_kg", "n_hops"}
    missing = required_cols - set(orders_df.columns)
    if missing:
        raise ValueError(f"orders_df missing required columns: {missing}")

    total_weight = orders_df["weight_kg"].sum()
    if total_weight == 0:
        return {"pass": True, "value": 0.0, "target": target}

    weighted = (orders_df["weight_kg"] * orders_df["n_hops"]).sum() / total_weight
    return {"pass": weighted <= target + 1e-9, "value": weighted, "target": target}


# ---------------------------------------------------------------------------
# S2 - Weighted average distance per kg within target
# ---------------------------------------------------------------------------

def check_s2_weighted_distance(orders_df: pd.DataFrame, target_km: float = 1300) -> dict:
    """orders_df: columns [weight_kg, routed_distance_km] — one row per
    served order."""
    required_cols = {"weight_kg", "routed_distance_km"}
    missing = required_cols - set(orders_df.columns)
    if missing:
        raise ValueError(f"orders_df missing required columns: {missing}")

    total_weight = orders_df["weight_kg"].sum()
    if total_weight == 0:
        return {"pass": True, "value": 0.0, "target_km": target_km}

    weighted = (orders_df["weight_kg"] * orders_df["routed_distance_km"]).sum() / total_weight
    return {"pass": weighted <= target_km + 1e-9, "value": weighted, "target_km": target_km}


# ---------------------------------------------------------------------------
# S3 - Single path between any hub pair
# ---------------------------------------------------------------------------

def check_s3_single_path_per_hub_pair(routes: list) -> dict:
    """routes: list of ordered node-sequences (each a list of node names),
    one per vehicle route or per candidate path in use.

    For every ordered pair of nodes (A, B) that co-occur (A before B) within
    ANY route, every route touching both A and B must traverse the exact
    same sub-path between them. Collapses to "at most one shipment-transfer
    path per hub pair" (Section 6, S3).
    """
    subpaths_by_pair = {}  # (A, B) -> set of distinct subpath tuples

    for route in routes:
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                a, b = route[i], route[j]
                subpath = tuple(route[i:j + 1])
                subpaths_by_pair.setdefault((a, b), set()).add(subpath)

    violations = {
        pair: sorted(paths) for pair, paths in subpaths_by_pair.items()
        if len(paths) > 1
    }

    return {
        "pass": len(violations) == 0,
        "n_hub_pairs_checked": len(subpaths_by_pair),
        "n_violations": len(violations),
        "violations": violations,
    }