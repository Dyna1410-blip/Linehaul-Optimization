"""
Reusable pass/fail checkers for hard constraints C1-C9 and soft targets S1-S3.
Used by both optimizer.py (to prune/select) and reports.py (to annotate
deliverables without recomputing).
"""


def check_c1_vehicle_capacity(phy_wt, vol_wt, phy_cap, vol_cap) -> bool:
    raise NotImplementedError


def check_c2_node_capacity(node_load_kg, processing_capacity_kg) -> bool:
    raise NotImplementedError


def check_c3_round_trip_distance(round_trip_km, limit_km) -> bool:
    raise NotImplementedError  # limit_km may be None (unbounded)


def check_c4_max_stops(n_intermediate_stops, max_stops=4) -> bool:
    raise NotImplementedError


def check_c5_directional(stop_sequence, dist_df) -> bool:
    raise NotImplementedError


def check_c8_detour_factor(routed_km, direct_km, detour_factor) -> bool:
    raise NotImplementedError


def check_c9_spillover(spilled_phy_kg, spilled_vol_kg, total_phy_kg, total_vol_kg,
                        limit_pct=0.12) -> dict:
    """Returns pass/fail separately for phy and vol."""
    raise NotImplementedError


def check_s1_weighted_hops(orders_df, target=0.2) -> dict:
    raise NotImplementedError


def check_s2_weighted_distance(orders_df, target_km=1300) -> dict:
    raise NotImplementedError


def check_s3_single_path_per_hub_pair(vehicle_routes_df) -> dict:
    raise NotImplementedError
