"""
Tests for src/optimizer.py — path selection, capacity-constrained daily
simulation, fleet-size search, and end-to-end run().
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from optimizer_milp import (
    select_paths, simulate_day_with_capacity, simulate_month,
    size_fleet, run,
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

ONE_WAY_DF = pd.DataFrame(columns=["origin", "destination", "pct"])


# --- select_paths ---

def test_select_paths_prefers_cheaper_option():
    # direct is short but let's make the "via B" option artificially cheap
    # by using a low-CPK, low-distance detour so it should win on cost/kg
    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 1000},
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 100},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 1000},
    ])
    chosen = select_paths(candidates, demand, VEHICLE_DF, HOP_COST_DF, ONE_WAY_DF, "17 FT")
    assert len(chosen) == 1
    # much shorter distance should win despite the added hop cost at B
    assert chosen.iloc[0]["path"] == "A|B|D"


def test_select_paths_skips_od_with_no_demand():
    candidates = pd.DataFrame([
        {"source_node": "X", "dest_node": "Y", "path": "X|Y", "total_distance_km": 100},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 1000},
    ])
    chosen = select_paths(candidates, demand, VEHICLE_DF, HOP_COST_DF, ONE_WAY_DF, "17 FT")
    assert chosen.empty


# --- simulate_day_with_capacity ---

def test_simulate_day_no_spillage_when_capacity_sufficient():
    day_demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
    ])
    chosen_paths = pd.DataFrame([{"source_node": "A", "dest_node": "D", "path": "A|D"}])
    result = simulate_day_with_capacity(day_demand, chosen_paths, VEHICLE_DF, "17 FT",
                                         fleet_capacity=5)
    assert result["spilled_phy_kg"] == 0.0
    assert len(result["routes"]) == 1


def test_simulate_day_spills_when_capacity_insufficient():
    # 2 legs, each needing 1 vehicle (small loads), but only 1 vehicle available total
    day_demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "B",
         "phy_wt_kg": 3000, "vol_wt_kg": 2500},
        {"date": pd.Timestamp("2026-07-01"), "source_node": "C", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 900},
    ])
    chosen_paths = pd.DataFrame([
        {"source_node": "A", "dest_node": "B", "path": "A|B"},
        {"source_node": "C", "dest_node": "D", "path": "C|D"},
    ])
    result = simulate_day_with_capacity(day_demand, chosen_paths, VEHICLE_DF, "17 FT",
                                         fleet_capacity=1)
    # biggest leg (A->B, 3000kg) should be served first; C->D (1000kg) spills
    assert result["spilled_phy_kg"] == 1000.0
    assert len(result["routes"]) == 1
    assert result["routes"][0]["start_node"] == "A"


def test_simulate_day_zero_capacity_spills_everything():
    day_demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 1000, "vol_wt_kg": 900},
    ])
    chosen_paths = pd.DataFrame([{"source_node": "A", "dest_node": "D", "path": "A|D"}])
    result = simulate_day_with_capacity(day_demand, chosen_paths, VEHICLE_DF, "17 FT",
                                         fleet_capacity=0)
    assert result["spilled_phy_kg"] == 1000.0
    assert len(result["routes"]) == 0


# --- simulate_month ---

def test_simulate_month_aggregates_across_days():
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
        {"date": pd.Timestamp("2026-07-02"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
    ])
    chosen_paths = pd.DataFrame([{"source_node": "A", "dest_node": "D", "path": "A|D"}])
    result = simulate_month(demand, chosen_paths, VEHICLE_DF, "17 FT", fleet_capacity=1)
    assert result["total_phy_kg"] == 8000
    assert result["total_spilled_phy_kg"] == 0.0  # 1 vehicle/day is enough here
    assert len(result["vehicle_routes"]) == 2  # 1 per day


# --- size_fleet ---

def test_size_fleet_finds_minimal_feasible_size():
    # peak day needs 2 vehicles (9000kg > one 5000kg vehicle), other days need 1
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 9000, "vol_wt_kg": 8000},  # peak -> needs 2 vehicles
        {"date": pd.Timestamp("2026-07-02"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},  # needs 1 vehicle
    ])
    chosen_paths = pd.DataFrame([{"source_node": "A", "dest_node": "D", "path": "A|D"}])

    # with a generous 12% limit, a fleet of 1 might be enough if peak-day
    # spillage stays under 12% of the MONTH's total weight (not that day's)
    result = size_fleet(demand, chosen_paths, VEHICLE_DF, "17 FT", limit_pct=0.12)
    assert result["fleet_size"] >= 1
    assert result["spillage"]["phy_spill_pct"] <= 0.12 + 1e-6
    assert result["spillage"]["vol_spill_pct"] <= 0.12 + 1e-6
    # fleet_size of 2 always trivially satisfies (0% spillage) -> search must
    # not return something worse than that
    assert result["fleet_size"] <= 2


def test_size_fleet_zero_demand_edge_case():
    demand = pd.DataFrame(columns=["date", "source_node", "dest_node", "phy_wt_kg", "vol_wt_kg"])
    chosen_paths = pd.DataFrame([{"source_node": "A", "dest_node": "D", "path": "A|D"}])
    result = size_fleet(demand, chosen_paths, VEHICLE_DF, "17 FT")
    assert result["fleet_size"] == 0


# --- run() end-to-end ---

def test_run_end_to_end():
    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 300},
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 280},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
        {"date": pd.Timestamp("2026-07-02"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
    ])
    result = run(candidates, demand, VEHICLE_DF, HOP_COST_DF, ONE_WAY_DF, vehicle_type="17 FT")

    assert result["vehicle_type"] == "17 FT"
    assert len(result["chosen_paths"]) == 1
    assert result["fleet_size"] >= 1
    assert result["spillage"]["phy_spill_pct"] <= 0.12 + 1e-6
    assert not result["allocation"].empty
    assert not result["vehicle_routes"].empty

    # C7 check: exactly one vehicle_type deployed, count should never
    # exceed fleet_size on any single day
    daily_counts = result["vehicle_routes"].groupby("date").size()
    assert (daily_counts <= result["fleet_size"]).all()


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in test_fns:
        fn()
    print(f"all {len(test_fns)} optimizer tests passed")
