"""
Tests for src/path_enumeration.py — includes a regression test for the
dest-as-intermediate-node bug (see fix log in the module docstring).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from path_enumeration import enumerate_paths_for_od, enumerate_all


def _self_loop_dist():
    """3-node network with explicit self-pairs (dist[X, X] = 0), matching
    the real distance matrix's shape (a full N x N grid)."""
    return {
        ("A", "A"): 0.0, ("B", "B"): 0.0, ("D", "D"): 0.0,
        ("A", "B"): 100.0, ("B", "A"): 100.0,
        ("A", "D"): 300.0, ("D", "A"): 300.0,
        ("B", "D"): 200.0, ("D", "B"): 200.0,
    }, ["A", "B", "D"]


def test_no_phantom_dest_hop():
    """Regression test: dest must never appear as an intermediate node.
    Before the fix this produced paths like A|B|D|D and A|D|D."""
    dist, nodes = _self_loop_dist()
    paths = enumerate_paths_for_od("A", "D", dist, nodes, detour_factor=1.5,
                                    max_intermediate_stops=4)
    for p in paths:
        path_nodes = p["path"]
        assert path_nodes[-1] == "D", "path must end at destination"
        assert "D" not in path_nodes[1:-1], (
            f"destination appears as an intermediate stop in {path_nodes}"
        )
        assert len(path_nodes) == len(set(path_nodes)), (
            f"path revisits a node: {path_nodes}"
        )

    # exactly the two genuine paths, no duplicates
    rendered = sorted("|".join(p["path"]) for p in paths)
    assert rendered == ["A|B|D", "A|D"], rendered


def test_direct_path_always_included_when_feasible():
    dist, nodes = _self_loop_dist()
    paths = enumerate_paths_for_od("A", "D", dist, nodes, detour_factor=1.0,
                                    max_intermediate_stops=4)
    assert any(p["path"] == ["A", "D"] for p in paths)


def test_detour_factor_prunes_infeasible_paths():
    dist, nodes = _self_loop_dist()
    # A->B->D = 300, direct A->D = 300, ratio 1.0 -> only feasible at factor >= 1.0
    tight = enumerate_paths_for_od("A", "D", dist, nodes, detour_factor=0.99,
                                    max_intermediate_stops=4)
    assert all(p["distance_km"] <= 0.99 * 300.0 + 1e-6 for p in tight)


def test_c4_max_intermediate_stops_respected():
    dist, nodes = _self_loop_dist()
    paths = enumerate_paths_for_od("A", "D", dist, nodes, detour_factor=5.0,
                                    max_intermediate_stops=0)
    # with 0 intermediate stops allowed, only the direct path can qualify
    assert all(p["num_intermediate"] == 0 for p in paths)


if __name__ == "__main__":
    test_no_phantom_dest_hop()
    test_direct_path_always_included_when_feasible()
    test_detour_factor_prunes_infeasible_paths()
    test_c4_max_intermediate_stops_respected()
    print("all path_enumeration tests passed")
