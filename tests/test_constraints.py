"""
Tests for src/constraints.py — one pass case and one violation case per
checker (C1-C9, S1-S3), using small synthetic tables.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from constraints import (
    check_c1_vehicle_capacity, check_c2_node_capacity,
    check_c3_round_trip_distance, check_c4_max_stops,
    check_c5_directional, check_c6_path_stability,
    check_c7_constant_daily_fleet, check_c8_detour_factor,
    check_c9_spillover, check_s1_weighted_hops,
    check_s2_weighted_distance, check_s3_single_path_per_hub_pair,
)


# --- C1 ---
def test_c1_pass_and_fail():
    ok = check_c1_vehicle_capacity(4000, 4000, 5000, 4463)
    assert ok["pass"]
    bad = check_c1_vehicle_capacity(6000, 4000, 5000, 4463)
    assert not bad["pass"] and not bad["phy_pass"] and bad["vol_pass"]


# --- C2 ---
def test_c2_pass_and_fail():
    ok = check_c2_node_capacity(40000, 50000)
    assert ok["pass"]
    bad = check_c2_node_capacity(60000, 50000)
    assert not bad["pass"]


# --- C3 ---
def test_c3_unbounded_and_bounded():
    unbounded = check_c3_round_trip_distance(500000, None)
    assert unbounded["pass"] and unbounded["unbounded"]
    ok = check_c3_round_trip_distance(1800, 2000)
    assert ok["pass"]
    bad = check_c3_round_trip_distance(2500, 2000)
    assert not bad["pass"]


# --- C4 ---
def test_c4_pass_and_fail():
    assert check_c4_max_stops(4, max_stops=4)["pass"]
    assert not check_c4_max_stops(5, max_stops=4)["pass"]


# --- C5 ---
def test_c5_pass_forward_progress():
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "D", "distance_km": 300},
        {"origin": "B", "destination": "D", "distance_km": 200},
        {"origin": "D", "destination": "D", "distance_km": 0},
    ])
    result = check_c5_directional(["A", "B", "D"], dist_df)
    assert result["pass"], result


def test_c5_fail_backtrack():
    # C -> D increases remaining distance to dest (should have gotten closer)
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "D", "distance_km": 300},
        {"origin": "C", "destination": "D", "distance_km": 500},  # further away
        {"origin": "D", "destination": "D", "distance_km": 0},
    ])
    result = check_c5_directional(["A", "C", "D"], dist_df)
    assert not result["pass"]
    assert len(result["violations"]) == 1


# --- C6 ---
def test_c6_pass_and_fail():
    stable = pd.DataFrame([
        {"date": "2026-07-01", "source_node": "A", "dest_node": "D", "portion_id": "p1", "path": "A|B|D"},
        {"date": "2026-07-02", "source_node": "A", "dest_node": "D", "portion_id": "p1", "path": "A|B|D"},
    ])
    assert check_c6_path_stability(stable)["pass"]

    unstable = pd.DataFrame([
        {"date": "2026-07-01", "source_node": "A", "dest_node": "D", "portion_id": "p1", "path": "A|B|D"},
        {"date": "2026-07-02", "source_node": "A", "dest_node": "D", "portion_id": "p1", "path": "A|D"},
    ])
    result = check_c6_path_stability(unstable)
    assert not result["pass"]
    assert result["n_violations"] == 1


# --- C7 ---
def test_c7_pass_and_fail():
    stable = pd.DataFrame([
        {"date": "2026-07-01", "vehicle_type": "17 FT", "n_deployed": 5},
        {"date": "2026-07-02", "vehicle_type": "17 FT", "n_deployed": 5},
    ])
    assert check_c7_constant_daily_fleet(stable)["pass"]

    unstable = pd.DataFrame([
        {"date": "2026-07-01", "vehicle_type": "17 FT", "n_deployed": 5},
        {"date": "2026-07-02", "vehicle_type": "17 FT", "n_deployed": 6},
    ])
    result = check_c7_constant_daily_fleet(unstable)
    assert not result["pass"]
    assert "17 FT" in result["violating_vehicle_types"]


# --- C8 ---
def test_c8_pass_and_fail():
    ok = check_c8_detour_factor(routed_km=330, direct_km=300, detour_factor=1.2)
    assert ok["pass"]
    bad = check_c8_detour_factor(routed_km=400, direct_km=300, detour_factor=1.2)
    assert not bad["pass"]


# --- C9 ---
def test_c9_pass_and_fail_independent_measures():
    ok = check_c9_spillover(spilled_phy_kg=1000, spilled_vol_kg=1000,
                             total_phy_kg=10000, total_vol_kg=10000)
    assert ok["pass"]
    # fails on vol only -> overall must fail (spec: independent check on both)
    mixed = check_c9_spillover(spilled_phy_kg=1000, spilled_vol_kg=2000,
                                total_phy_kg=10000, total_vol_kg=10000)
    assert mixed["phy_pass"] and not mixed["vol_pass"] and not mixed["pass"]


# --- S1 ---
def test_s1_pass_and_fail():
    orders = pd.DataFrame([
        {"weight_kg": 8000, "n_hops": 0},
        {"weight_kg": 2000, "n_hops": 1},
    ])
    # weighted = (8000*0 + 2000*1)/10000 = 0.2 -> exactly at target, passes
    result = check_s1_weighted_hops(orders, target=0.2)
    assert result["pass"], result

    heavy_hop = pd.DataFrame([
        {"weight_kg": 5000, "n_hops": 1},
        {"weight_kg": 5000, "n_hops": 1},
    ])
    result2 = check_s1_weighted_hops(heavy_hop, target=0.2)
    assert not result2["pass"]


# --- S2 ---
def test_s2_pass_and_fail():
    orders = pd.DataFrame([
        {"weight_kg": 5000, "routed_distance_km": 1000},
        {"weight_kg": 5000, "routed_distance_km": 1000},
    ])
    assert check_s2_weighted_distance(orders, target_km=1300)["pass"]

    far = pd.DataFrame([
        {"weight_kg": 5000, "routed_distance_km": 2000},
        {"weight_kg": 5000, "routed_distance_km": 2000},
    ])
    assert not check_s2_weighted_distance(far, target_km=1300)["pass"]


# --- S3 ---
def test_s3_pass_consistent_routes():
    routes = [
        ["SC_1", "SC_2", "SC_4"],
        ["SC_1", "SC_2", "SC_4"],  # same sub-path SC_1->SC_4 both times
    ]
    result = check_s3_single_path_per_hub_pair(routes)
    assert result["pass"], result


def test_s3_fail_inconsistent_routes():
    routes = [
        ["SC_1", "SC_2", "SC_4"],
        ["SC_1", "SC_3", "SC_4"],  # different sub-path between SC_1 and SC_4
    ]
    result = check_s3_single_path_per_hub_pair(routes)
    assert not result["pass"]
    assert ("SC_1", "SC_4") in result["violations"]


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in test_fns:
        fn()
    print(f"all {len(test_fns)} constraints tests passed")