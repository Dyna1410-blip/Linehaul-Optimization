"""
Tests for src/cost_model.py — covers the four cost components (Section 2)
and one-way lane pricing (Section 3) against small synthetic tables.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from cost_model import (
    round_trip_cost, trip_cost, hop_cost, touch_cost,
    price_path_estimate, lookup_one_way_factor,
)

VEHICLE_DF = pd.DataFrame([
    {"vehicle_type": "17 FT", "phy_cap_kg": 5000, "vol_cap_kg": 4463,
     "fixed_cost": 1145.0, "per_km_cost": 18.70, "round_trip_km_limit": 1000},
])

HOP_COST_DF = pd.DataFrame([
    {"node": "A", "cpk": 1.0},
    {"node": "B", "cpk": 2.0},
    {"node": "D", "cpk": 1.5},
])

ONE_WAY_DF = pd.DataFrame([
    {"origin": "A", "destination": "B", "pct": 0.6},
])


def test_round_trip_cost_doubles_distance():
    # fixed 1145 + per_km 18.70 * (2*100) = 1145 + 3740 = 4885
    cost = round_trip_cost("17 FT", 100.0, VEHICLE_DF)
    assert abs(cost - 4885.0) < 1e-6, cost


def test_trip_cost_no_one_way_lane_charges_full_round_trip():
    # A->D is not in ONE_WAY_DF -> full round trip cost charged
    result = trip_cost("17 FT", 100.0, "A", "D", VEHICLE_DF, ONE_WAY_DF)
    assert result["one_way_pct_applied"] is None
    assert abs(result["total"] - 4885.0) < 1e-6, result


def test_trip_cost_one_way_lane_applies_pct():
    # A->B is in ONE_WAY_DF at 0.6 -> 60% of round-trip cost
    result = trip_cost("17 FT", 50.0, "A", "B", VEHICLE_DF, ONE_WAY_DF)
    full_round_trip = 1145.0 + 18.70 * 100.0  # 2*50
    expected = full_round_trip * 0.6
    assert result["one_way_pct_applied"] == 0.6
    assert abs(result["total"] - expected) < 1e-6, result


def test_hop_cost_vs_touch_cost_ratio():
    h = hop_cost("B", 1000.0, HOP_COST_DF)
    t = touch_cost("B", 1000.0, HOP_COST_DF)
    assert abs(h - 2000.0) < 1e-6, h          # 2.0 cpk * 1000 kg
    assert abs(t - 1000.0) < 1e-6, t          # half of hop cost
    assert abs(t - 0.5 * h) < 1e-9


def test_one_way_factor_lookup_missing_lane_returns_none():
    assert lookup_one_way_factor("X", "Y", ONE_WAY_DF) is None
    assert lookup_one_way_factor("A", "B", ONE_WAY_DF) == 0.6


def test_price_path_estimate_direct_path_has_no_hop_cost():
    # direct A->D, no intermediate stops
    path_row = pd.Series({
        "source_node": "A", "dest_node": "D", "path": "A|D",
        "total_distance_km": 100.0,
    })
    result = price_path_estimate(path_row, "17 FT", 1000.0,
                                  VEHICLE_DF, HOP_COST_DF, ONE_WAY_DF)
    assert result["hop"] == 0.0
    assert result["intermediate_nodes"] == []
    assert abs(result["total"] - 4885.0) < 1e-6, result


def test_price_path_estimate_via_intermediate_adds_hop_cost():
    # A->B->D, B is the intermediate hop
    path_row = pd.Series({
        "source_node": "A", "dest_node": "D", "path": "A|B|D",
        "total_distance_km": 120.0,
    })
    result = price_path_estimate(path_row, "17 FT", 1000.0,
                                  VEHICLE_DF, HOP_COST_DF, ONE_WAY_DF)
    expected_hop = 2.0 * 1000.0  # cpk[B] * weight
    trip_total = 1145.0 + 18.70 * (2 * 120.0)
    assert abs(result["hop"] - expected_hop) < 1e-6
    assert result["intermediate_nodes"] == ["B"]
    assert abs(result["total"] - (trip_total + expected_hop)) < 1e-6, result


if __name__ == "__main__":
    test_round_trip_cost_doubles_distance()
    test_trip_cost_no_one_way_lane_charges_full_round_trip()
    test_trip_cost_one_way_lane_applies_pct()
    test_hop_cost_vs_touch_cost_ratio()
    test_one_way_factor_lookup_missing_lane_returns_none()
    test_price_path_estimate_direct_path_has_no_hop_cost()
    test_price_path_estimate_via_intermediate_adds_hop_cost()
    print("all cost_model tests passed")
