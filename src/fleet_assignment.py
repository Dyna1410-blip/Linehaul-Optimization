"""
Given chosen paths (post-optimizer) and demand, assign orders to specific
vehicles/legs — including load splitting where an order spans multiple
vehicles on the same leg or across legs.

Produces the granular (leg, order, vehicle) records needed for report 7.3,
and the per-vehicle stop sequences / loads needed for report 7.4.
"""


def assign_orders_to_vehicles(chosen_paths_df, demand_df, vehicle_df):
    raise NotImplementedError


def build_vehicle_routes(assignment_df, dist_df):
    raise NotImplementedError
