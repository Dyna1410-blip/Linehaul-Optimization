"""
Tests for src/fleet_assignment.py — bin packing correctness (including
forced order splitting) and end-to-end assign_fleet() behavior.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from fleet_assignment import (
    choose_shortest_path_per_od, explode_orders_to_legs,
    pack_orders_into_vehicles, assign_fleet,
)

VEHICLE_DF = pd.DataFrame([
    {"vehicle_type": "17 FT", "phy_cap_kg": 5000, "vol_cap_kg": 4463,
     "fixed_cost": 1145.0, "per_km_cost": 18.70, "round_trip_km_limit": 1000},
])


def test_choose_shortest_path_per_od():
    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 300},
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 250},
    ])
    chosen = choose_shortest_path_per_od(candidates)
    assert len(chosen) == 1
    assert chosen.iloc[0]["path"] == "A|B|D"


def test_explode_orders_to_legs_direct():
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 1000},
    ])
    chosen_paths = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D"},
    ])
    exploded = explode_orders_to_legs(demand, chosen_paths)
    assert len(exploded) == 1
    assert exploded.iloc[0]["leg_from"] == "A"
    assert exploded.iloc[0]["leg_to"] == "D"


def test_explode_orders_to_legs_via_intermediate():
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 1000},
    ])
    chosen_paths = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|B|D"},
    ])
    exploded = explode_orders_to_legs(demand, chosen_paths)
    assert len(exploded) == 2  # A->B, B->D
    legs = list(zip(exploded["leg_from"], exploded["leg_to"]))
    assert legs == [("A", "B"), ("B", "D")]
    # full weight carried on BOTH legs
    assert (exploded["phy_wt_kg"] == 1000).all()


def test_explode_missing_path_skips_order():
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "X", "dest_node": "Y",
         "phy_wt_kg": 1000, "vol_wt_kg": 1000},
    ])
    chosen_paths = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D"},
    ])
    exploded = explode_orders_to_legs(demand, chosen_paths)
    assert exploded.empty


def test_pack_orders_single_vehicle_no_split_needed():
    orders = [
        {"order_id": "o1", "phy_wt_kg": 2000, "vol_wt_kg": 1800},
        {"order_id": "o2", "phy_wt_kg": 1000, "vol_wt_kg": 900},
    ]
    vehicles = pack_orders_into_vehicles(orders, "17 FT", VEHICLE_DF)
    assert len(vehicles) == 1
    assert len(vehicles[0]["orders"]) == 2
    assert abs(vehicles[0]["phy_load_kg"] - 3000) < 1e-6


def test_pack_orders_forces_split_across_vehicles():
    # single order of 8000 kg > 5000 kg vehicle capacity -> must split
    orders = [{"order_id": "big_order", "phy_wt_kg": 8000, "vol_wt_kg": 7000}]
    vehicles = pack_orders_into_vehicles(orders, "17 FT", VEHICLE_DF)
    assert len(vehicles) == 2, f"expected split into 2 vehicles, got {len(vehicles)}"

    total_phy = sum(v["phy_load_kg"] for v in vehicles)
    total_vol = sum(v["vol_load_kg"] for v in vehicles)
    assert abs(total_phy - 8000) < 1e-6
    assert abs(total_vol - 7000) < 1e-6

    # every vehicle stayed within capacity (C1)
    for v in vehicles:
        assert v["phy_load_kg"] <= 5000 + 1e-6
        assert v["vol_load_kg"] <= 4463 + 1e-6

    # the split order appears on both vehicles with the SAME order_id
    order_ids_seen = set()
    for v in vehicles:
        for o in v["orders"]:
            order_ids_seen.add(o["order_id"])
    assert order_ids_seen == {"big_order"}


def test_pack_orders_vol_binding_forces_split_even_if_phy_fits():
    # phy fits in one vehicle (4000 < 5000 cap) but vol doesn't (8000 > 4463 cap)
    orders = [{"order_id": "bulky", "phy_wt_kg": 4000, "vol_wt_kg": 8000}]
    vehicles = pack_orders_into_vehicles(orders, "17 FT", VEHICLE_DF)
    assert len(vehicles) == 2
    for v in vehicles:
        assert v["vol_load_kg"] <= 4463 + 1e-6


def test_assign_fleet_end_to_end():
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 900},
        {"date": pd.Timestamp("2026-07-02"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1200, "vol_wt_kg": 1100},
    ])
    chosen_paths = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|B|D"},
    ])
    result = assign_fleet(demand, chosen_paths, VEHICLE_DF, vehicle_type="17 FT")
    allocation = result["allocation"]
    routes = result["vehicle_routes"]

    # 2 days x 2 legs (A->B, B->D) = 4 vehicle-leg groups, 1 vehicle each (small loads)
    assert len(routes) == 4
    assert set(routes["start_node"] + "->" + routes["end_node"]) == {"A->B", "B->D"}

    # every allocation row references a real vehicle_id from routes
    assert set(allocation["vehicle_id"]).issubset(set(routes["vehicle_id"]))

    # weight conservation: total phy_wt_kg allocated per leg matches demand
    day1_a_to_b = allocation[
        (allocation["date"] == pd.Timestamp("2026-07-01")) &
        (allocation["leg_from"] == "A") & (allocation["leg_to"] == "B")
    ]
    assert abs(day1_a_to_b["phy_wt_kg"].sum() - 1000) < 1e-6


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in test_fns:
        fn()
    print(f"all {len(test_fns)} fleet_assignment tests passed")
