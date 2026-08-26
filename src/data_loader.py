"""
Load and validate the six raw inputs (Section 4), normalize schemas, and
return clean DataFrames for downstream modules.

Column names are normalized defensively (case/whitespace-insensitive
matching against a few known aliases) because the exact casing/spacing in
the raw files (e.g. "Per KM cost (Fixed)", "%age") is easy to get subtly
wrong. If a required column can't be found, the loader raises with the
full list of columns it actually found, so a mismatch is fast to diagnose
instead of failing silently or crashing deep in cost_model.

Outputs:
- demand_df:   date (datetime), source_node, dest_node, phy_wt_kg, vol_wt_kg,
               total_shipments, detour_factor
- node_df:     node, processing_capacity_kg, processing_time_hrs
- dist_df:     origin, destination, distance_km, time_hrs
- vehicle_df:  vehicle_type, phy_cap_kg, vol_cap_kg, fixed_cost, per_km_cost,
               round_trip_km_limit (NaN/None = unbounded)
- hop_cost_df: node, cpk
- one_way_df:  origin, destination, pct (fraction 0-1, share of round-trip
               cost charged for a one-way trip on that lane)
"""

import os
import pandas as pd
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find_column(df: pd.DataFrame, candidates: list, required_for: str) -> str:
    """Case/whitespace-insensitive match against a list of acceptable column
    names. Raises with the actual columns found if nothing matches."""
    norm_map = {c.strip().lower().replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ", "")
        if key in norm_map:
            return norm_map[key]
    raise ValueError(
        f"Could not find a column for '{required_for}'. "
        f"Looked for any of {candidates}. "
        f"Actual columns in file: {list(df.columns)}"
    )


def _raw_path(cfg: dict, key: str) -> str:
    return os.path.join(cfg["paths"]["raw_dir"], cfg["paths"][key])


def load_demand(cfg: dict) -> pd.DataFrame:
    path = _raw_path(cfg, "demand_file")
    df = _normalize_columns(pd.read_csv(path))

    col_date = _find_column(df, ["date"], "date")
    col_source = _find_column(df, ["source_node", "source"], "source_node")
    col_dest = _find_column(df, ["dest_node", "destination_node", "dest"], "dest_node")
    col_phy = _find_column(df, ["total_Phy_wt", "total_phy_wt", "phy_wt"], "total_Phy_wt")
    col_vol = _find_column(df, ["total_Vol_wt", "total_vol_wt", "vol_wt"], "total_Vol_wt")
    col_ship = _find_column(df, ["total_shipments", "shipments"], "total_shipments")
    col_detour = _find_column(df, ["detour_factor"], "detour_factor")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[col_date], format="%d-%m-%Y", errors="coerce"),
        "source_node": df[col_source].astype(str).str.strip(),
        "dest_node": df[col_dest].astype(str).str.strip(),
        "phy_wt_kg": pd.to_numeric(df[col_phy], errors="coerce"),
        "vol_wt_kg": pd.to_numeric(df[col_vol], errors="coerce"),
        "total_shipments": pd.to_numeric(df[col_ship], errors="coerce"),
        "detour_factor": pd.to_numeric(df[col_detour], errors="coerce"),
    })

    if out["date"].isna().any():
        n_bad = out["date"].isna().sum()
        raise ValueError(f"{n_bad} rows in {path} have an unparseable date "
                          f"(expected DD-MM-YYYY).")
    return out


def load_nodes(cfg: dict) -> pd.DataFrame:
    path = _raw_path(cfg, "node_file")
    df = _normalize_columns(pd.read_csv(path))

    col_node = _find_column(df, ["node"], "node")
    col_cap = _find_column(df, ["processing_capacity_kgs", "processing_capacity_kg"],
                            "processing_capacity_kgs")
    col_time = _find_column(df, ["processing_time_hrs"], "processing_time_hrs")

    return pd.DataFrame({
        "node": df[col_node].astype(str).str.strip(),
        "processing_capacity_kg": pd.to_numeric(df[col_cap], errors="coerce"),
        "processing_time_hrs": pd.to_numeric(df[col_time], errors="coerce"),
    })


def load_distances(cfg: dict) -> pd.DataFrame:
    path = _raw_path(cfg, "distance_file")
    df = _normalize_columns(pd.read_csv(path))

    col_o = _find_column(df, ["origin"], "origin")
    col_d = _find_column(df, ["destination"], "destination")
    col_dist = _find_column(df, ["distance_km"], "distance_km")
    col_time = _find_column(df, ["time_hrs"], "time_hrs")

    return pd.DataFrame({
        "origin": df[col_o].astype(str).str.strip(),
        "destination": df[col_d].astype(str).str.strip(),
        "distance_km": pd.to_numeric(df[col_dist], errors="coerce"),
        "time_hrs": pd.to_numeric(df[col_time], errors="coerce"),
    })


def load_vehicles(cfg: dict) -> pd.DataFrame:
    path = _raw_path(cfg, "vehicle_file")
    df = _normalize_columns(pd.read_csv(path))

    col_type = _find_column(df, ["Vehicle Type", "vehicle_type", "type"], "Vehicle Type")
    col_phy = _find_column(
        df, ["Phy Cap(Kgs)", "Phy Cap (kg)", "Phy Cap", "phy_cap_kg"], "Phy Cap(Kgs)")
    col_vol = _find_column(
        df, ["Vol Cap(Kgs)", "Vol Cap (kg)", "Vol Cap", "vol_cap_kg"], "Vol Cap(Kgs)")
    col_fixed = _find_column(df, ["Fixed cost", "fixed_cost"], "Fixed cost")
    col_perkm = _find_column(df, ["Per KM cost (Fixed)", "Per-km cost", "per_km_cost"],
                              "Per KM cost (Fixed)")
    col_limit = _find_column(
        df, ["Round Trip KM Range", "Round-trip km limit", "round_trip_km_limit"],
        "Round Trip KM Range")

    out = pd.DataFrame({
        "vehicle_type": df[col_type].astype(str).str.strip(),
        "phy_cap_kg": pd.to_numeric(df[col_phy], errors="coerce"),
        "vol_cap_kg": pd.to_numeric(df[col_vol], errors="coerce"),
        "fixed_cost": pd.to_numeric(df[col_fixed], errors="coerce"),
        "per_km_cost": pd.to_numeric(df[col_perkm], errors="coerce"),
    })
    # "unbounded" / blank / "-" all mean no limit -> NaN
    limit_raw = df[col_limit].astype(str).str.strip().str.lower()
    limit_num = pd.to_numeric(df[col_limit], errors="coerce")
    out["round_trip_km_limit"] = limit_num.where(
        ~limit_raw.isin(["unbounded", "", "-", "nan", "none"]), other=pd.NA
    )

    # Bonus column present in the real file but not documented in the
    # problem statement (Section 5, C4 states a flat cap of 4 for every
    # type). If present, surface it so constraints.py can use a
    # per-vehicle-type cap instead of a single global constant, in case
    # it differs from 4 for some types.
    try:
        col_stops = _find_column(
            df, ["max_intermediate_stops", "max intermediate stops"],
            "max_intermediate_stops")
        out["max_intermediate_stops"] = pd.to_numeric(df[col_stops], errors="coerce")
    except ValueError:
        out["max_intermediate_stops"] = pd.NA  # not present -> caller falls back to config default

    return out


def load_hop_costs(cfg: dict) -> pd.DataFrame:
    """Per guidance: ignore the Period column entirely. The CPK file's
    node coverage is scattered across different Period labels rather than
    being period-varying rates for the same node, so hop cost is treated
    as a flat per-node rate regardless of month."""
    path = _raw_path(cfg, "hop_cost_file")
    df = _normalize_columns(pd.read_csv(path))

    col_loc = _find_column(df, ["Location", "node"], "Location")
    col_cpk = _find_column(df, ["CPK", "cpk"], "CPK")

    out = pd.DataFrame({
        "node": df[col_loc].astype(str).str.strip(),
        "cpk": pd.to_numeric(df[col_cpk], errors="coerce"),
    })

    # If the same node appears more than once (e.g. under different Period
    # labels) with DIFFERENT CPK values, that's worth knowing about rather
    # than silently keeping whichever row happens to come first.
    dupes = out.groupby("node")["cpk"].nunique()
    conflicting = dupes[dupes > 1]
    if not conflicting.empty:
        print(
            f"WARNING: {path} has multiple different CPK values for the "
            f"same node (period ignored per guidance): "
            f"{conflicting.index.tolist()}. Keeping the first value seen "
            f"per node — review data/raw for these nodes if that's wrong."
        )

    return out.drop_duplicates(subset="node", keep="first")


def load_one_way_lanes(cfg: dict) -> pd.DataFrame:
    path = _raw_path(cfg, "one_way_lane_file")
    df = _normalize_columns(pd.read_excel(path))

    col_o = _find_column(df, ["origin", "source_node", "source", "from"], "origin")
    col_d = _find_column(df, ["destination", "dest_node", "dest", "to"], "destination")
    col_pct = _find_column(df, ["%age", "pct", "percentage"], "%age")

    pct_raw = pd.to_numeric(df[col_pct], errors="coerce")
    # Normalize: values like 60 mean 60% -> 0.6; values already <=1 are left as-is.
    pct = pct_raw.where(pct_raw <= 1.0, other=pct_raw / 100.0)

    return pd.DataFrame({
        "origin": df[col_o].astype(str).str.strip(),
        "destination": df[col_d].astype(str).str.strip(),
        "pct": pct,
    })


def load_all(cfg: dict) -> dict:
    """Convenience wrapper returning all six tables as a dict of DataFrames."""
    return {
        "demand": load_demand(cfg),
        "nodes": load_nodes(cfg),
        "distances": load_distances(cfg),
        "vehicles": load_vehicles(cfg),
        "hop_costs": load_hop_costs(cfg),
        "one_way_lanes": load_one_way_lanes(cfg),
    }