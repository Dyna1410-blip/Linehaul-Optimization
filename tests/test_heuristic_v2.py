"""
Tests for src/heuristic_v2.py — mixed-fleet vehicle type selection,
hop-vs-touch decision (including a regression test for the amortization
bug caught during development), and end-to-end weight conservation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from heuristic_v2 import (
    pick_vehicle_type_for_leg, estimate_path_mode_and_cost,
    run_iterative_heuristic, simulate_month_v2,
)

VEHICLE_DF = pd.DataFrame([
    {"vehicle_type": "32 FT (Multi Axle)", "phy_cap_kg": 18000, "vol_cap_kg": 12920,
     "fixed_cost": 3364, "per_km_cost": 33.17, "round_trip_km_limit": None},
    {"vehicle_type": "Mahindra Pickup", "phy_cap_kg": 1200, "vol_cap_kg": 1472,
     "fixed_cost": 563, "per_km_cost": 13.16, "round_trip_km_limit": 600},
])
ONE_WAY_DF = pd.DataFrame(columns=["origin", "destination", "pct"])
DIST_DF = pd.DataFrame([
    {"origin": "A", "destination": "B", "distance_km": 100, "time_hrs": 2},
    {"origin": "B", "destination": "D", "distance_km": 100, "time_hrs": 2},
])
PATH_ROW = pd.Series({"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 200})


def test_mixed_fleet_thin_leg_prefers_small_vehicle():
    choice = pick_vehicle_type_for_leg(800, 100, VEHICLE_DF)
    assert choice == "Mahindra Pickup", choice


def test_mixed_fleet_thick_leg_prefers_large_vehicle():
    choice = pick_vehicle_type_for_leg(50000, 100, VEHICLE_DF)
    assert choice == "32 FT (Multi Axle)", choice


def test_hop_touch_expensive_hop_favors_through():
    # cpk=50 is deliberately extreme so the hop cost dominates regardless
    # of leg-sharing amortization — a moderate hop cost (e.g. 5.0) can
    # legitimately still favor leg-mode if the leg is heavily shared, since
    # BOTH hop cost AND amortized dispatch savings genuinely trade off
    # against each other (see the cheap-hop test below for that case).
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "B", "cpk": 50.0}, {"node": "D", "cpk": 1.5},
    ])
    leg_volumes = {("A", "B"): 50000, ("B", "D"): 50000}
    result = estimate_path_mode_and_cost(PATH_ROW, 4000, {}, leg_volumes, VEHICLE_DF,
                                          hop_cost_df, ONE_WAY_DF, DIST_DF, "32 FT (Multi Axle)")
    assert result["mode"] == "through", result


def test_hop_touch_cheap_hop_with_shared_volume_favors_leg():
    """Regression test: without amortizing leg cost over the leg's REALIZED
    shared volume, this incorrectly always picked 'through' regardless of
    hop cost, because it compared 'one dedicated dispatch' (through) vs
    'two dedicated dispatches, one per leg, for this order alone' (leg) —
    an unfair comparison that ignores leg-mode's real advantage: sharing
    a leg's vehicle with OTHER OD flows amortizes its fixed cost."""
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "B", "cpk": 0.01}, {"node": "D", "cpk": 1.5},
    ])
    leg_volumes = {("A", "B"): 50000, ("B", "D"): 50000}
    result = estimate_path_mode_and_cost(PATH_ROW, 4000, {}, leg_volumes, VEHICLE_DF,
                                          hop_cost_df, ONE_WAY_DF, DIST_DF, "32 FT (Multi Axle)")
    assert result["mode"] == "leg", result


def test_end_to_end_weight_conservation():
    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 300},
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 200},
        {"source_node": "C", "dest_node": "D", "path": "C|D", "total_distance_km": 150},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4000, "vol_wt_kg": 3500},
        {"date": pd.Timestamp("2026-07-02"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 4200, "vol_wt_kg": 3600},
        {"date": pd.Timestamp("2026-07-01"), "source_node": "C", "dest_node": "D",
         "phy_wt_kg": 800, "vol_wt_kg": 700},
        {"date": pd.Timestamp("2026-07-02"), "source_node": "C", "dest_node": "D",
         "phy_wt_kg": 900, "vol_wt_kg": 800},
    ])
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "B", "cpk": 0.3},
        {"node": "C", "cpk": 1.0}, {"node": "D", "cpk": 1.5},
    ])
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "D", "distance_km": 300, "time_hrs": 5},
        {"origin": "A", "destination": "B", "distance_km": 100, "time_hrs": 2},
        {"origin": "B", "destination": "D", "distance_km": 100, "time_hrs": 2},
        {"origin": "C", "destination": "D", "distance_km": 150, "time_hrs": 3},
    ])

    chosen, leg_vehicle_types = run_iterative_heuristic(candidates, demand, VEHICLE_DF, hop_cost_df,
                                                         ONE_WAY_DF, dist_df, n_iterations=3, verbose=False)
    result = simulate_month_v2(demand, chosen, VEHICLE_DF, hop_cost_df, ONE_WAY_DF, dist_df,
                                leg_vehicle_types)

    assert abs(result["allocation"]["phy_wt_kg"].sum() - demand["phy_wt_kg"].sum()) < 1e-6
    assert result["total_cost"] > 0
    assert not result["fleet_size"].empty
    # C1 sanity: no vehicle instance exceeds its own type's capacity
    routes = result["vehicle_routes"]
    for row in routes.itertuples(index=False):
        cap = VEHICLE_DF.loc[VEHICLE_DF["vehicle_type"] == row.vehicle_type, "phy_cap_kg"].iloc[0]
        assert row.phy_load_kg <= cap + 1e-6


def test_mixed_fleet_actually_used_in_simulation():
    """Regression test for the mixed-fleet bug: leg-mode dispatches were
    silently falling back to the largest vehicle type regardless of the
    per-leg type computed during selection, because that per-leg mapping
    never made it into chosen_paths_df (only through-mode paths carry a
    single vehicle_type column)."""
    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 200},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 800, "vol_wt_kg": 700},
    ])
    vehicle_df = pd.DataFrame([
        {"vehicle_type": "32 FT (Multi Axle)", "phy_cap_kg": 18000, "vol_cap_kg": 12920,
         "fixed_cost": 3364, "per_km_cost": 33.17, "round_trip_km_limit": 250},
        {"vehicle_type": "Mahindra Pickup", "phy_cap_kg": 1200, "vol_cap_kg": 1472,
         "fixed_cost": 563, "per_km_cost": 13.16, "round_trip_km_limit": 250},
    ])
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "B", "cpk": 0.3}, {"node": "D", "cpk": 1.5},
    ])
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "B", "distance_km": 100, "time_hrs": 2},
        {"origin": "B", "destination": "D", "distance_km": 100, "time_hrs": 2},
    ])

    chosen, leg_vehicle_types = run_iterative_heuristic(candidates, demand, vehicle_df, hop_cost_df,
                                                         ONE_WAY_DF, dist_df, n_iterations=3, verbose=False)
    assert chosen["mode"].iloc[0] == "leg"
    result = simulate_month_v2(demand, chosen, vehicle_df, hop_cost_df, ONE_WAY_DF, dist_df,
                                leg_vehicle_types)
    # this thin 800kg load should use the small pickup, not the 18000kg-cap truck
    used_types = set(result["vehicle_routes"]["vehicle_type"])
    assert used_types == {"Mahindra Pickup"}, used_types


def test_node_capacity_violation_gets_resolved():
    """Regression test for the C2 enforcement gap: confirms a genuine
    violation gets detected AND corrected via re-routing to a safe
    alternate path, not just silently reported."""
    from heuristic_v2 import resolve_node_capacity_violations

    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 300},
        {"source_node": "A", "dest_node": "D", "path": "A|B|D", "total_distance_km": 150},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp(f"2026-07-{d:02d}"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 20000, "vol_wt_kg": 18000}
        for d in range(1, 4)
    ])
    vehicle_df = pd.DataFrame([
        {"vehicle_type": "32 FT (Multi Axle)", "phy_cap_kg": 18000, "vol_cap_kg": 12920,
         "fixed_cost": 3364, "per_km_cost": 33.17, "round_trip_km_limit": 250},
    ])
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "B", "cpk": 0.3}, {"node": "D", "cpk": 1.5},
    ])
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "D", "distance_km": 300, "time_hrs": 5},
        {"origin": "A", "destination": "B", "distance_km": 75, "time_hrs": 1},
        {"origin": "B", "destination": "D", "distance_km": 75, "time_hrs": 1},
    ])
    node_df = pd.DataFrame([
        {"node": "A", "processing_capacity_kg": 100000, "processing_time_hrs": 2},
        {"node": "B", "processing_capacity_kg": 15000, "processing_time_hrs": 2},
        {"node": "D", "processing_capacity_kg": 100000, "processing_time_hrs": 2},
    ])

    chosen, leg_vtypes = run_iterative_heuristic(candidates, demand, vehicle_df, hop_cost_df,
                                                  ONE_WAY_DF, dist_df, n_iterations=2, verbose=False)
    resolved = resolve_node_capacity_violations(
        candidates, demand, chosen, leg_vtypes, vehicle_df, hop_cost_df,
        ONE_WAY_DF, dist_df, node_df, max_rounds=5, verbose=False,
    )
    assert resolved["remaining_violations"].empty
    assert resolved["n_swaps_made"] >= 1
    assert resolved["chosen_paths"]["path"].iloc[0] == "A|D"  # swapped off the tight-node path


def test_spill_to_fit_resolves_unavoidable_violation():
    """A is the ORIGIN for both flows, so rerouting can never avoid it —
    only spillage can fix an over-capacity origin/destination node."""
    from heuristic_v2 import (
        resolve_node_capacity_violations, find_node_capacity_violations,
        spill_to_fit_remaining_violations,
    )

    candidates = pd.DataFrame([
        {"source_node": "A", "dest_node": "D", "path": "A|D", "total_distance_km": 100},
        {"source_node": "A", "dest_node": "C", "path": "A|C", "total_distance_km": 80},
    ])
    demand = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "D",
         "phy_wt_kg": 8000, "vol_wt_kg": 7000},
        {"date": pd.Timestamp("2026-07-01"), "source_node": "A", "dest_node": "C",
         "phy_wt_kg": 5000, "vol_wt_kg": 4000},
    ])
    vehicle_df = pd.DataFrame([
        {"vehicle_type": "32 FT (Multi Axle)", "phy_cap_kg": 18000, "vol_cap_kg": 12920,
         "fixed_cost": 3364, "per_km_cost": 33.17, "round_trip_km_limit": None},
    ])
    hop_cost_df = pd.DataFrame([
        {"node": "A", "cpk": 1.0}, {"node": "C", "cpk": 1.0}, {"node": "D", "cpk": 1.5},
    ])
    dist_df = pd.DataFrame([
        {"origin": "A", "destination": "D", "distance_km": 100, "time_hrs": 2},
        {"origin": "A", "destination": "C", "distance_km": 80, "time_hrs": 1},
    ])
    node_df = pd.DataFrame([
        {"node": "A", "processing_capacity_kg": 12000, "processing_time_hrs": 2},  # total demand 13000 > 12000
        {"node": "C", "processing_capacity_kg": 100000, "processing_time_hrs": 2},
        {"node": "D", "processing_capacity_kg": 100000, "processing_time_hrs": 2},
    ])

    chosen, leg_vtypes = run_iterative_heuristic(candidates, demand, vehicle_df, hop_cost_df,
                                                  ONE_WAY_DF, dist_df, n_iterations=2, verbose=False)
    resolved = resolve_node_capacity_violations(
        candidates, demand, chosen, leg_vtypes, vehicle_df, hop_cost_df,
        ONE_WAY_DF, dist_df, node_df, max_rounds=3, verbose=False,
    )
    assert not resolved["remaining_violations"].empty  # rerouting genuinely can't fix this

    spilled = spill_to_fit_remaining_violations(
        demand, resolved["chosen_paths"], resolved["leg_vehicle_types"], vehicle_df, hop_cost_df,
        ONE_WAY_DF, dist_df, node_df, resolved["remaining_violations"], verbose=False,
    )
    final_violations = find_node_capacity_violations(spilled["result"]["vehicle_routes"], node_df)
    assert final_violations.empty
    assert spilled["total_spilled_phy_kg"] > 0
    assert spilled["within_c9_limit"]  # 1000kg spilled / 13000kg total = 7.7%, under 12%


if __name__ == "__main__":
    test_fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in test_fns:
        fn()
    print(f"all {len(test_fns)} heuristic_v2 tests passed")