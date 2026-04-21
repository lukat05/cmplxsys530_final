"""
summaries.py — Task 2.3: Tree and Epidemic Summary Statistics
=============================================================

Provides two main functions:

  compute_tree_summaries(T)
      Scalar summary statistics for the sampled transmission genealogy T
      (a networkx DiGraph produced by Task 2.2's prune_to_sampled).

  compute_epidemic_summaries(Y)
      Scalar summary statistics for the observed epidemic data Y
      (an ObservedEpidemicData produced by Task 2.1's observation module).

Tree summary statistics implemented:
  - Branching-time distribution (mean, variance, Q25, Q75)
  - Tree depth (max root-to-leaf path in time units)
  - Cherry count
  - Sackin index
  - Colless index (binary trees only)
  - Ladder length

Epidemic summary statistics:
  - Total outbreak size
  - Peak incidence
  - Time to peak
  - Final epidemic size

Expected sampled-genealogy format (networkx.DiGraph, from Task 2.2):
  Node attributes:
    - 'host_id'        : int
    - 'infection_time' : float
    - 'is_sampled'     : bool  (True for observed leaf nodes)
    - 'covariate'      : float (present if is_sampled)

  Edges are directed: parent → child, with optional 'time' attribute.
  Internal nodes may be unsampled ancestors retained to connect the
  sampled leaves.  After pruning, degree-2 internal nodes are compressed,
  so the tree is typically resolved (binary or multifurcating).

Author: Luka Todorovic
Course: CMPLXSYS 530 / EPI 638, Winter 2026
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np

# We import the data container from observation.py so the epidemic
# summaries function has a typed input.
try:
    from observation import ObservedEpidemicData
except ImportError:
    # Allow running from project root as well
    from luka.observation import ObservedEpidemicData


# ===================================================================
# Helper utilities
# ===================================================================

def _find_roots(T: nx.DiGraph) -> List[int]:
    """Return nodes with in-degree 0 (roots of the transmission forest)."""
    return [n for n in T.nodes() if T.in_degree(n) == 0]


def _get_leaves(T: nx.DiGraph) -> List[int]:
    """Return nodes with out-degree 0 (leaves of the tree)."""
    return [n for n in T.nodes() if T.out_degree(n) == 0]


def _node_time(T: nx.DiGraph, node: int) -> float:
    """Return the infection_time attribute for a node."""
    return float(T.nodes[node]["infection_time"])


def _leaf_depths_time(T: nx.DiGraph) -> Dict[int, float]:
    """
    Compute the depth of every leaf, measured in *time units*
    (infection_time of leaf minus infection_time of the tree root).

    For a forest (multiple roots), each leaf's depth is relative to
    its own root.
    """
    roots = _find_roots(T)
    leaves = set(_get_leaves(T))
    depths: Dict[int, float] = {}

    for root in roots:
        root_time = _node_time(T, root)
        # BFS / DFS from root
        for node in nx.descendants(T, root) | {root}:
            if node in leaves:
                depths[node] = _node_time(T, node) - root_time

    return depths


def _leaf_depths_topological(T: nx.DiGraph) -> Dict[int, int]:
    """
    Compute the depth of every leaf, measured in *edge count*
    (number of edges from root to leaf).

    Used for the Sackin index.
    """
    roots = _find_roots(T)
    leaves = set(_get_leaves(T))
    depths: Dict[int, int] = {}

    for root in roots:
        # BFS gives shortest path lengths (= depth in a tree)
        for node, dist in nx.single_source_shortest_path_length(T, root).items():
            if node in leaves:
                depths[node] = dist

    return depths


def _count_descendant_leaves(T: nx.DiGraph, node: int, leaf_set: Set[int]) -> int:
    """
    Count how many leaves (from leaf_set) are in the subtree rooted at *node*.
    """
    if node in leaf_set:
        return 1
    count = 0
    for child in T.successors(node):
        count += _count_descendant_leaves(T, child, leaf_set)
    return count


# ===================================================================
# Tree summary statistics
# ===================================================================

def compute_tree_summaries(T: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute scalar summary statistics for the sampled transmission
    genealogy T.

    Parameters
    ----------
    T : nx.DiGraph
        Sampled genealogy (from Task 2.2).  Nodes must have
        'infection_time' attributes.  Leaves are nodes with
        out-degree 0.

    Returns
    -------
    dict
        Keys and their meanings:

        Branching-time distribution
        ---------------------------
        'bt_mean'   : float — mean of internal-node infection times
        'bt_var'    : float — variance of internal-node infection times
        'bt_q25'    : float — 25th percentile
        'bt_q75'    : float — 75th percentile
        'bt_times'  : np.ndarray — raw vector of internal-node times

        Tree shape
        ----------
        'tree_depth_time' : float — max root-to-leaf depth (time units)
        'tree_depth_edges': int   — max root-to-leaf depth (edge count)
        'cherry_count'    : int   — # of internal nodes whose children
                                    are all leaves
        'sackin_index'    : int   — sum of leaf depths (edge count)
        'colless_index'   : float or None — sum |L_left - L_right| over
                                    internal nodes (None if non-binary)
        'ladder_length'   : int   — longest chain of consecutive
                                    degree-2 internal nodes on any
                                    root-to-leaf path
        'n_leaves'        : int   — number of leaves
        'n_internal'      : int   — number of internal nodes
        'is_binary'       : bool  — True if every internal node has
                                    exactly 2 children
    """
    if T.number_of_nodes() == 0:
        return _empty_tree_summaries()

    leaves = set(_get_leaves(T))
    internal = set(T.nodes()) - leaves
    n_leaves = len(leaves)
    n_internal = len(internal)

    # ------------------------------------------------------------------
    # 1. Branching-time distribution (internal node infection times)
    # ------------------------------------------------------------------
    if n_internal > 0:
        bt_times = np.array([_node_time(T, n) for n in internal])
        bt_mean = float(np.mean(bt_times))
        bt_var = float(np.var(bt_times))
        bt_q25 = float(np.percentile(bt_times, 25))
        bt_q75 = float(np.percentile(bt_times, 75))
    else:
        # Degenerate: tree is a single leaf
        bt_times = np.array([])
        bt_mean = bt_var = bt_q25 = bt_q75 = 0.0

    # ------------------------------------------------------------------
    # 2. Tree depth
    # ------------------------------------------------------------------
    leaf_depths_t = _leaf_depths_time(T)
    leaf_depths_e = _leaf_depths_topological(T)

    tree_depth_time = max(leaf_depths_t.values()) if leaf_depths_t else 0.0
    tree_depth_edges = max(leaf_depths_e.values()) if leaf_depths_e else 0

    # ------------------------------------------------------------------
    # 3. Cherry count
    #    An internal node is a "cherry" if ALL of its children are leaves.
    # ------------------------------------------------------------------
    cherry_count = 0
    for node in internal:
        children = list(T.successors(node))
        if len(children) == 2 and all(c in leaves for c in children):
            cherry_count += 1

    # ------------------------------------------------------------------
    # 4. Sackin index: sum of leaf depths (in edge count)
    # ------------------------------------------------------------------
    sackin_index = sum(leaf_depths_e.values()) if leaf_depths_e else 0

    # ------------------------------------------------------------------
    # 5. Colless index (defined only for fully binary trees)
    #    For each internal node: |n_left_leaves - n_right_leaves|
    # ------------------------------------------------------------------
    is_binary = all(T.out_degree(n) == 2 for n in internal) if n_internal > 0 else True

    if is_binary and n_internal > 0:
        colless_index = 0.0
        for node in internal:
            children = list(T.successors(node))
            left_leaves = _count_descendant_leaves(T, children[0], leaves)
            right_leaves = _count_descendant_leaves(T, children[1], leaves)
            colless_index += abs(left_leaves - right_leaves)
    else:
        colless_index = None  # undefined for non-binary trees

    # ------------------------------------------------------------------
    # 6. Ladder length
    #    A "degree-2 internal node" has exactly 1 child that is also
    #    internal (i.e., out-degree 1 if we ignore the parent edge, but
    #    in the directed-tree sense: out-degree == 1 AND the child is
    #    also internal).
    #
    #    We find the maximum run of consecutive such nodes along any
    #    root-to-leaf path.
    # ------------------------------------------------------------------
    ladder_length = _compute_ladder_length(T, leaves, internal)

    return {
        # Branching-time distribution
        "bt_mean": bt_mean,
        "bt_var": bt_var,
        "bt_q25": bt_q25,
        "bt_q75": bt_q75,
        "bt_times": bt_times,
        # Tree shape
        "tree_depth_time": tree_depth_time,
        "tree_depth_edges": tree_depth_edges,
        "cherry_count": cherry_count,
        "sackin_index": sackin_index,
        "colless_index": colless_index,
        "ladder_length": ladder_length,
        # Metadata
        "n_leaves": n_leaves,
        "n_internal": n_internal,
        "is_binary": is_binary,
    }


def _empty_tree_summaries() -> Dict[str, Any]:
    """Return a dict of summary statistics for an empty tree."""
    return {
        "bt_mean": 0.0,
        "bt_var": 0.0,
        "bt_q25": 0.0,
        "bt_q75": 0.0,
        "bt_times": np.array([]),
        "tree_depth_time": 0.0,
        "tree_depth_edges": 0,
        "cherry_count": 0,
        "sackin_index": 0,
        "colless_index": None,
        "ladder_length": 0,
        "n_leaves": 0,
        "n_internal": 0,
        "is_binary": True,
    }


def _compute_ladder_length(
    T: nx.DiGraph,
    leaves: Set[int],
    internal: Set[int],
) -> int:
    """
    Compute the maximum ladder length in the tree.

    A ladder segment at a node occurs when that node has out-degree 1
    (a single child), which is the hallmark of a "comb-like" region
    in the transmission tree.  We walk every root-to-leaf path and
    track the longest consecutive run of out-degree-1 internal nodes.
    """
    if not internal:
        return 0

    roots = _find_roots(T)
    max_ladder = 0

    def _dfs(node: int, current_run: int) -> None:
        nonlocal max_ladder
        children = list(T.successors(node))
        n_children = len(children)

        if n_children == 0:
            # Leaf — finalize current run
            max_ladder = max(max_ladder, current_run)
            return

        # A node contributes to a ladder if it has exactly 1 child
        # (degree-2 in the undirected sense: 1 parent + 1 child)
        if n_children == 1 and node in internal:
            new_run = current_run + 1
        else:
            # Run broken at a branching/multifurcating node —
            # record the accumulated run before resetting
            max_ladder = max(max_ladder, current_run)
            new_run = 0

        for child in children:
            _dfs(child, new_run)

    for root in roots:
        _dfs(root, 0)

    return max_ladder


# ===================================================================
# Epidemic summary statistics
# ===================================================================

def compute_epidemic_summaries(Y: ObservedEpidemicData) -> Dict[str, Any]:
    """
    Compute scalar summary statistics from the observed epidemic data Y.

    Parameters
    ----------
    Y : ObservedEpidemicData
        Output of observation.observe() or observation.build_observed_epidemic_data().

    Returns
    -------
    dict
        'total_outbreak_size' : int   — total sampled cases
        'peak_incidence'      : float — max incidence in any time bin
        'time_to_peak'        : float — midpoint of the peak time bin
        'final_epidemic_size' : int   — total eventually infected (sampled)
        'epidemic_duration'   : float — time span from first to last
                                        infection among sampled hosts
        'mean_generation'     : float — mean generation depth of sampled
                                        hosts (if available)
    """
    if Y.total_observed_cases == 0:
        return {
            "total_outbreak_size": 0,
            "peak_incidence": 0.0,
            "time_to_peak": 0.0,
            "final_epidemic_size": 0,
            "epidemic_duration": 0.0,
            "mean_generation": 0.0,
        }

    inf_times = [h.infection_time for h in Y.sampled_hosts]
    generations = [h.generation for h in Y.sampled_hosts]

    return {
        "total_outbreak_size": Y.total_observed_cases,
        "peak_incidence": Y.peak_incidence,
        "time_to_peak": Y.time_to_peak,
        "final_epidemic_size": Y.final_epidemic_size,
        "epidemic_duration": float(max(inf_times) - min(inf_times)),
        "mean_generation": float(np.mean(generations)),
    }


# ===================================================================
# Self-test / demonstration
# ===================================================================

def _make_demo_binary_tree() -> nx.DiGraph:
    r"""
    Build a small hand-crafted binary tree for verifiable testing.

    Structure (infection times in parentheses)::

              0 (t=0)
             / \
           1(1)  2(2)
          / \     / \
        3(3) 4(4) 5(5) 6(6)

    Leaves: {3, 4, 5, 6}
    Internal: {0, 1, 2}
    """
    T = nx.DiGraph()
    nodes = {
        0: {"host_id": 0, "infection_time": 0.0, "is_sampled": False, "covariate": 0.5},
        1: {"host_id": 1, "infection_time": 1.0, "is_sampled": False, "covariate": 0.3},
        2: {"host_id": 2, "infection_time": 2.0, "is_sampled": False, "covariate": 0.7},
        3: {"host_id": 3, "infection_time": 3.0, "is_sampled": True, "covariate": 0.2},
        4: {"host_id": 4, "infection_time": 4.0, "is_sampled": True, "covariate": 0.4},
        5: {"host_id": 5, "infection_time": 5.0, "is_sampled": True, "covariate": 0.6},
        6: {"host_id": 6, "infection_time": 6.0, "is_sampled": True, "covariate": 0.8},
    }
    for nid, attrs in nodes.items():
        T.add_node(nid, **attrs)
    T.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    return T


def _make_demo_comb_tree() -> nx.DiGraph:
    r"""
    Build a comb-shaped (caterpillar) tree for ladder-length testing.

    Structure::

        0 (t=0)
        |
        1 (t=1)
        |
        2 (t=2)
       / \
      3(3) 4(4)

    Ladder at nodes 0→1 (both have out-degree 1): length = 2
    Leaves: {3, 4}
    Internal: {0, 1, 2}
    """
    T = nx.DiGraph()
    for i in range(5):
        T.add_node(i, host_id=i, infection_time=float(i),
                   is_sampled=(i >= 3), covariate=0.5)
    T.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4)])
    return T


def _make_demo_multifurcating_tree() -> nx.DiGraph:
    r"""
    A tree with a trifurcation to test Colless = None.

    ::

        0
       /|\
      1  2  3
    """
    T = nx.DiGraph()
    for i in range(4):
        T.add_node(i, host_id=i, infection_time=float(i),
                   is_sampled=(i > 0), covariate=0.5)
    T.add_edges_from([(0, 1), (0, 2), (0, 3)])
    return T


if __name__ == "__main__":
    print("=" * 60)
    print("Task 2.3 — Tree Summary Statistics: self-test")
    print("=" * 60)

    # ---- Test 1: Balanced binary tree ----
    print("\n--- Test 1: Balanced binary tree ---")
    T_binary = _make_demo_binary_tree()
    s = compute_tree_summaries(T_binary)

    print(f"  Leaves         : {s['n_leaves']}  (expect 4)")
    assert s["n_leaves"] == 4
    print(f"  Internal       : {s['n_internal']}  (expect 3)")
    assert s["n_internal"] == 3
    print(f"  Is binary      : {s['is_binary']}  (expect True)")
    assert s["is_binary"] is True

    # Branching times = internal node times = [0, 1, 2]
    print(f"  bt_mean        : {s['bt_mean']:.2f}  (expect 1.00)")
    assert abs(s["bt_mean"] - 1.0) < 1e-9
    print(f"  bt_var         : {s['bt_var']:.4f}  (expect 0.6667)")
    assert abs(s["bt_var"] - 2.0/3.0) < 1e-9

    # Tree depth: deepest leaf is at time 6, root at 0 → depth = 6
    print(f"  tree_depth_time: {s['tree_depth_time']:.1f}  (expect 6.0)")
    assert abs(s["tree_depth_time"] - 6.0) < 1e-9
    # Edge depth: root→child→leaf = 2 edges
    print(f"  tree_depth_edge: {s['tree_depth_edges']}  (expect 2)")
    assert s["tree_depth_edges"] == 2

    # Cherries: nodes 1 and 2 each have 2 leaf children
    print(f"  cherry_count   : {s['cherry_count']}  (expect 2)")
    assert s["cherry_count"] == 2

    # Sackin: leaf depths in edges = [2, 2, 2, 2], sum = 8
    print(f"  sackin_index   : {s['sackin_index']}  (expect 8)")
    assert s["sackin_index"] == 8

    # Colless: node 0 → |2-2|=0, node 1 → |1-1|=0, node 2 → |1-1|=0 → total 0
    print(f"  colless_index  : {s['colless_index']}  (expect 0.0)")
    assert s["colless_index"] == 0.0

    # Ladder: no out-degree-1 internal nodes → ladder = 0
    print(f"  ladder_length  : {s['ladder_length']}  (expect 0)")
    assert s["ladder_length"] == 0

    print("  ✓ All assertions passed")

    # ---- Test 2: Comb tree (ladder) ----
    print("\n--- Test 2: Comb tree (ladder) ---")
    T_comb = _make_demo_comb_tree()
    s2 = compute_tree_summaries(T_comb)

    print(f"  Leaves         : {s2['n_leaves']}  (expect 2)")
    assert s2["n_leaves"] == 2
    print(f"  Internal       : {s2['n_internal']}  (expect 3)")
    assert s2["n_internal"] == 3

    # Ladder: 0→1 are degree-1 internal nodes; then 2 has degree 2
    # So the run is: node 0 (deg-1) → node 1 (deg-1) → node 2 (deg-2, breaks)
    # Ladder length = 2
    print(f"  ladder_length  : {s2['ladder_length']}  (expect 2)")
    assert s2["ladder_length"] == 2

    # Binary: node 0 has 1 child, so not fully binary
    print(f"  is_binary      : {s2['is_binary']}  (expect False)")
    assert s2["is_binary"] is False

    # Colless should be None for non-binary
    print(f"  colless_index  : {s2['colless_index']}  (expect None)")
    assert s2["colless_index"] is None

    # Sackin: leaf 3 depth=3, leaf 4 depth=3 → sum=6
    print(f"  sackin_index   : {s2['sackin_index']}  (expect 6)")
    assert s2["sackin_index"] == 6

    print("  ✓ All assertions passed")

    # ---- Test 3: Multifurcating tree ----
    print("\n--- Test 3: Multifurcating tree ---")
    T_multi = _make_demo_multifurcating_tree()
    s3 = compute_tree_summaries(T_multi)

    print(f"  is_binary      : {s3['is_binary']}  (expect False)")
    assert s3["is_binary"] is False
    print(f"  colless_index  : {s3['colless_index']}  (expect None)")
    assert s3["colless_index"] is None
    print(f"  cherry_count   : {s3['cherry_count']}  (expect 0)")
    # Node 0 has 3 children, all leaves — but a cherry requires exactly 2
    # children that are leaves. The spec says "exactly two sampled leaf
    # descendants" — node 0 has 3, so it's NOT a cherry.
    assert s3["cherry_count"] == 0

    print("  ✓ All assertions passed")

    # ---- Test 4: Epidemic summaries ----
    print("\n--- Test 4: Epidemic summaries ---")
    # Reuse the demo tree builder from observation.py
    try:
        from observation import observe, _make_demo_tree
    except ImportError:
        from luka.observation import observe, _make_demo_tree

    demo_epi_tree = _make_demo_tree(n=200, seed=99)
    Y = observe(demo_epi_tree, pi=0.5, rng=np.random.default_rng(42))
    epi_s = compute_epidemic_summaries(Y)

    print(f"  total_outbreak_size : {epi_s['total_outbreak_size']}")
    print(f"  peak_incidence      : {epi_s['peak_incidence']:.1f}")
    print(f"  time_to_peak        : {epi_s['time_to_peak']:.2f}")
    print(f"  final_epidemic_size : {epi_s['final_epidemic_size']}")
    print(f"  epidemic_duration   : {epi_s['epidemic_duration']:.2f}")
    print(f"  mean_generation     : {epi_s['mean_generation']:.2f}")

    assert epi_s["total_outbreak_size"] > 0
    assert epi_s["peak_incidence"] > 0
    assert epi_s["epidemic_duration"] > 0
    print("  ✓ All assertions passed")

    # ---- Test 5: Empty tree ----
    print("\n--- Test 5: Empty tree ---")
    s_empty = compute_tree_summaries(nx.DiGraph())
    assert s_empty["n_leaves"] == 0
    assert s_empty["sackin_index"] == 0
    print("  ✓ Empty-tree edge case handled")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)
