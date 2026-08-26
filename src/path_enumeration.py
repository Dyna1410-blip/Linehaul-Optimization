"""
Path Enumeration for Middle-Mile Line-Haul Network Optimization
==================================================================

Enumerates all feasible consolidation paths for every OD pair, subject to:
  - C4: at most 4 intermediate stops (path length <= 6 nodes incl. origin/dest)
  - C5: directional routing - each successive stop must not increase the
        remaining distance to the destination (no backtracking / zig-zag)
  - C8: detour factor - total routed distance <= detour_factor * direct distance

This is the upstream step that turns a 24-node complete graph + 552 OD pairs
into a tractable candidate-path set for the downstream path-selection MILP.
Without this pruning, raw multi-commodity flow variables over a 24-node graph
would be intractable at this scale; enumerating up front lets the MILP pick
path *shares* per OD pair from a small candidate list instead.

Usage:
    python path_enumeration.py \
        --distance-matrix one_to_all_distance_matrix_full.csv \
        --demand monthly_clustered_loads_combined.csv \
        --output candidate_paths.csv

Output columns:
    source_node, dest_node, path (pipe-separated node sequence),
    num_intermediate_stops, total_distance_km, direct_distance_km,
    detour_ratio, detour_factor_limit

Fix log:
    2026-08-25 - dest-as-intermediate bug. The distance matrix is a full
    N x N grid (576 rows = 24^2), which includes self-pairs dist[X, X] = 0.
    The DFS extension loop didn't exclude nxt == dest, so it could "hop
    into" dest as if it were an intermediate stop and keep recursing from
    there (dist[dest, dest] == 0 trivially passes the C5 and C8 checks),
    producing malformed duplicate paths like A -> B -> D -> D with a
    phantom extra stop. Fixed by excluding nxt == dest from the extension
    loop; reaching dest is handled exclusively by the "finish" check.
    Verified against a synthetic 3- and 4-node network.
"""

import argparse
import csv
from collections import defaultdict


def load_distance_matrix(path):
    """
    Load one_to_all_distance_matrix_full.csv.
    Expected columns: origin, destination, key, time_hrs, distance_km
    Returns:
        dist: dict[(origin, dest)] -> distance_km (float)
        nodes: sorted list of unique node names
    """
    dist = {}
    nodes = set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            o = row["origin"].strip()
            d = row["destination"].strip()
            dist[(o, d)] = float(row["distance_km"])
            nodes.add(o)
            nodes.add(d)
    return dist, sorted(nodes)


def load_od_pairs(demand_path):
    """
    Load monthly_clustered_loads_combined.csv and extract the distinct set of
    OD pairs with their detour_factor. detour_factor is constant per OD pair
    across the month (per the problem statement), so we just take the first
    occurrence of each (source_node, dest_node) pair.

    Expected columns: date, source_node, dest_node, total_Phy_wt, total_Vol_wt,
                       total_shipments, detour_factor
    Returns:
        od_pairs: dict[(source, dest)] -> detour_factor (float)
    """
    od_pairs = {}
    with open(demand_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["source_node"].strip()
            d = row["dest_node"].strip()
            key = (s, d)
            if key not in od_pairs:
                od_pairs[key] = float(row["detour_factor"])
    return od_pairs


def enumerate_paths_for_od(origin, dest, dist, nodes, detour_factor,
                            max_intermediate_stops=4):
    """
    Branch-and-bound DFS enumerating all feasible consolidation paths from
    origin to dest.

    Pruning applied INSIDE the recursion (not post-hoc filtering):
      - C5 forward progress: only extend to a candidate `nxt` if
            dist[nxt, dest] <= dist[current, dest]
        (each hop must not increase the remaining distance to destination)
      - C8 detour cap, applied as a lower-bound completion check:
            accumulated_distance + dist[current, dest] <= budget
        i.e. even the best-case direct finish from here must stay in budget,
        or the branch is dead and we cut it immediately.
      - C4 depth cap: recursion depth (intermediate stops) <= max_intermediate_stops

    Returns a list of dicts, one per feasible path:
        {"path": [node, ...], "distance_km": float, "num_intermediate": int}
    """
    direct_dist = dist.get((origin, dest))
    if direct_dist is None:
        return []  # no direct distance on file - can't compute detour budget

    budget = detour_factor * direct_dist
    results = []

    def lower_bound_complete(current, accumulated):
        # cheapest possible way to finish from `current` is to go direct to dest
        finish = dist.get((current, dest))
        if finish is None:
            return None
        return accumulated + finish

    def dfs(current, path, visited, accumulated):
        # Try finishing directly at dest right now (this covers the direct
        # path with 0 intermediate stops, and every intermediate completion)
        leg = dist.get((current, dest))
        if leg is not None:
            total = accumulated + leg
            if total <= budget + 1e-9:
                results.append({
                    "path": path + [dest],
                    "distance_km": round(total, 3),
                    "num_intermediate": len(path) - 1,  # path includes origin
                })

        # Stop extending if we've hit the intermediate-stop cap
        if len(path) - 1 >= max_intermediate_stops:
            return

        for nxt in nodes:
            if nxt == current or nxt == origin or nxt == dest or nxt in visited:
                continue
            # nxt == dest is excluded deliberately: reaching dest is handled
            # exclusively by the "finish" check above. If the distance matrix
            # includes self-pairs (dist[dest, dest] == 0, as a full N x N
            # matrix typically does), letting dest be treated as an
            # intermediate node lets the DFS "hop into" dest and keep
            # recursing from there, since dist[dest, dest] == 0 trivially
            # passes both the C5 forward-progress check and the C8
            # lower-bound prune. That produces malformed duplicate paths
            # like A -> B -> D -> D with a phantom extra stop.
            leg_dist = dist.get((current, nxt))
            if leg_dist is None:
                continue

            # --- C5: forward progress toward destination ---
            dist_next_to_dest = dist.get((nxt, dest))
            dist_curr_to_dest = dist.get((current, dest))
            if dist_next_to_dest is None or dist_curr_to_dest is None:
                continue
            if dist_next_to_dest > dist_curr_to_dest + 1e-9:
                continue  # would increase remaining distance -> backtracking

            new_accumulated = accumulated + leg_dist

            # --- C8: lower-bound prune ---
            lb = lower_bound_complete(nxt, new_accumulated)
            if lb is None or lb > budget + 1e-9:
                continue  # even best-case finish blows the detour budget

            visited.add(nxt)
            dfs(nxt, path + [nxt], visited, new_accumulated)
            visited.remove(nxt)

    dfs(origin, [origin], {origin}, 0.0)
    return results


def enumerate_all(dist, nodes, od_pairs, max_intermediate_stops=4, top_k=None):
    """
    Run enumeration for every OD pair. Returns a flat list of row dicts
    ready to write to CSV.

    top_k: if set, keep only the top_k shortest-distance candidate paths per
    OD pair. This matters in practice: loose detour_factor lanes (3.0/4.0/5.0)
    can generate thousands of near-duplicate candidate paths, which would
    blow up the downstream path-selection MILP for no real benefit - most of
    those extra candidates are marginal variations on the same few genuinely
    distinct consolidation routings. Always keeps the direct path if it
    exists, even if it wouldn't otherwise make the cut (it won't, since it's
    always the shortest option, but stated for clarity).
    """
    all_rows = []
    skipped = []
    for (o, d), detour_factor in od_pairs.items():
        direct = dist.get((o, d))
        if direct is None:
            skipped.append((o, d))
            continue
        paths = enumerate_paths_for_od(o, d, dist, nodes, detour_factor,
                                        max_intermediate_stops)
        if top_k is not None and len(paths) > top_k:
            paths = sorted(paths, key=lambda p: p["distance_km"])[:top_k]
        for p in paths:
            all_rows.append({
                "source_node": o,
                "dest_node": d,
                "path": "|".join(p["path"]),
                "num_intermediate_stops": p["num_intermediate"],
                "total_distance_km": p["distance_km"],
                "direct_distance_km": round(direct, 3),
                "detour_ratio": round(p["distance_km"] / direct, 4) if direct > 0 else None,
                "detour_factor_limit": detour_factor,
            })
    return all_rows, skipped


def write_output(rows, out_path):
    fieldnames = ["source_node", "dest_node", "path", "num_intermediate_stops",
                  "total_distance_km", "direct_distance_km", "detour_ratio",
                  "detour_factor_limit"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_summary(rows, od_pairs, skipped):
    by_od = defaultdict(int)
    for r in rows:
        by_od[(r["source_node"], r["dest_node"])] += 1

    counts = list(by_od.values())
    n_od = len(od_pairs)
    n_with_paths = len(by_od)

    print(f"OD pairs in demand file:        {n_od}")
    print(f"OD pairs with >=1 valid path:   {n_with_paths}")
    print(f"OD pairs with NO valid path:    {n_od - n_with_paths - len(skipped)}"
          f" (infeasible under current detour_factor / C5 combo)")
    print(f"OD pairs skipped (no distance): {len(skipped)}")
    if counts:
        print(f"Candidate paths per OD pair:    "
              f"min={min(counts)}, max={max(counts)}, "
              f"avg={sum(counts)/len(counts):.1f}")
    print(f"Total candidate paths generated: {len(rows)}")

    direct_only = sum(1 for r in rows if r["num_intermediate_stops"] == 0)
    print(f"Of which pure direct (0 stops):  {direct_only}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-matrix", required=True,
                         help="Path to one_to_all_distance_matrix_full.csv")
    parser.add_argument("--demand", required=True,
                         help="Path to monthly_clustered_loads_combined.csv")
    parser.add_argument("--output", default="candidate_paths.csv",
                         help="Output CSV path (default: candidate_paths.csv)")
    parser.add_argument("--max-intermediate-stops", type=int, default=4,
                         help="C4 cap on intermediate stops (default: 4)")
    parser.add_argument("--top-k", type=int, default=30,
                         help="Keep only the K shortest candidate paths per OD "
                              "pair (default: 30). Set to 0 to disable and keep "
                              "all feasible paths (can be very large on loose "
                              "detour-factor lanes - see README).")
    args = parser.parse_args()
    top_k = args.top_k if args.top_k and args.top_k > 0 else None

    print("Loading distance matrix...")
    dist, nodes = load_distance_matrix(args.distance_matrix)
    print(f"  {len(nodes)} nodes, {len(dist)} directed OD distance entries")

    print("Loading demand / OD pairs...")
    od_pairs = load_od_pairs(args.demand)
    print(f"  {len(od_pairs)} distinct OD pairs")

    print("Enumerating feasible paths (C4 + C5 + C8)...")
    rows, skipped = enumerate_all(dist, nodes, od_pairs, args.max_intermediate_stops, top_k)

    print("Writing output...")
    write_output(rows, args.output)
    print(f"  Written to {args.output}")
    print()
    print_summary(rows, od_pairs, skipped)


if __name__ == "__main__":
    main()