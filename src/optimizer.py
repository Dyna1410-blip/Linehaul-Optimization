"""
Core decision layer: choose one path per OD-pair split (C6: path stability —
same path every day), size a constant daily fleet (C7), and decide which
load spills (C9, <=12% each of phy/vol weight).

Orchestrates: data_loader -> path_enumeration -> cost_model -> constraints
-> fleet_assignment, then reports.

TODO: pick formulation (MILP via pulp/OR-tools, or a heuristic/column-
generation approach given 552 OD pairs x candidate paths x 9 vehicle types
x 31 days is likely too large for naive MILP).
"""


def run(cfg: dict):
    raise NotImplementedError
