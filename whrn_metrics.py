"""
whrn_metrics.py

Python port of your MATLAB WHRN .m files:

- calc_dist_stats.m
- calculate_gammaM.m
- calculate_rhoM.m
- calculate_tauM.m
- calculate_global_whrn.m (+ _old)
- calculate_inner_class_whrn.m (+ _old)
- calculate_cross_class_whrn.m (+ _old)

Design choice:
- Uses networkx.Graph for the network.
- Edge attributes:
    * "weight"    = connection strength (from W_ij)
    * "distance"  = 1/(weight + epsilon)  (edge COST used for shortest paths)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_1d_array(x: Sequence[Any]) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _default_nodelist(G: nx.Graph) -> List[Any]:
    # Stable order. If nodes are 1..N this matches MATLAB-style.
    return sorted(G.nodes())


def adjacency_weight_matrix(
    G: nx.Graph,
    nodelist: Optional[Sequence[Any]] = None,
    *,
    weight: str = "weight",
) -> np.ndarray:
    """
    Weighted adjacency matrix aligned to nodelist.
    Missing edges -> 0.
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)
    return nx.to_numpy_array(G, nodelist=nodes, weight=weight, dtype=float)


def all_pairs_shortest_path_matrix(
    G: nx.Graph,
    nodelist: Optional[Sequence[Any]] = None,
    *,
    weight: str = "distance",
) -> np.ndarray:
    """
    All-pairs shortest path matrix using Dijkstra (positive weights),
    returning np.inf for unreachable pairs.
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    N = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    dist = np.full((N, N), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)

    for src, lengths in nx.all_pairs_dijkstra_path_length(G, weight=weight):
        i = idx[src]
        for dst, d in lengths.items():
            j = idx[dst]
            dist[i, j] = float(d)

    return dist


# ---------------------------------------------------------------------
# MATLAB buildG() equivalent (helper to create a graph with Distance)
# ---------------------------------------------------------------------

def build_graph(
    W: np.ndarray,
    node_property: Optional[Sequence[Any]] = None,
    epsilon: float = 1e-8,
    *,
    directed: bool = False,
    assume_symmetric: bool = True,
    nodes_start_at_1: bool = True,
) -> nx.Graph:
    """
    Build a (di)graph from an NxN weighted adjacency matrix W.

    Edge attributes:
        weight    : W_ij
        distance  : 1/(weight + epsilon)

    Node attributes:
        property  : node_property[i]

    Parameters
    ----------
    W : (N,N) array
        Weighted adjacency; 0 => no edge.
    node_property : optional, length N
    epsilon : float
        Small constant to avoid division by zero when computing distance.
    directed : bool
        If True, create nx.DiGraph and use all i!=j entries.
    assume_symmetric : bool
        If False (and directed=False), uses max(W, W.T) to symmetrize.
    nodes_start_at_1 : bool
        If True, nodes are labeled 1..N to match MATLAB indexing.
        If False, nodes are labeled 0..N-1.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square NxN matrix.")
    N = W.shape[0]

    if node_property is not None:
        node_property = np.asarray(node_property)
        if node_property.shape[0] != N:
            raise ValueError("node_property must have length N (same as W.shape[0]).")

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()

    nodes = list(range(1, N + 1)) if nodes_start_at_1 else list(range(N))
    G.add_nodes_from(nodes)

    if node_property is not None:
        for i, node in enumerate(nodes):
            G.nodes[node]["property"] = node_property[i]

    def add_edge(i: int, j: int, w: float) -> None:
        if w == 0 or not np.isfinite(w):
            return
        dist = 1.0 / (float(w) + float(epsilon))
        G.add_edge(nodes[i], nodes[j], weight=float(w), distance=float(dist))

    if directed:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                add_edge(i, j, W[i, j])
    else:
        if not assume_symmetric:
            W = np.maximum(W, W.T)
        for i in range(N):
            for j in range(i + 1, N):
                add_edge(i, j, W[i, j])

    return G


# ---------------------------------------------------------------------
# MATLAB: calc_dist_stats.m
# ---------------------------------------------------------------------

def calc_dist_stats(data: Sequence[float], prefix: str) -> Tuple[np.ndarray, List[str]]:
    """
    Computes 8 statistical descriptors:
        [Min, Q1, Median, Q3, Max, IQR, Mean, Std]

    Returns
    -------
    stats : (8,) array
    names : list[str] length 8
    """
    suffixes = ["Min", "Q1", "Median", "Q3", "Max", "IQR", "Mean", "Std"]
    names = [f"{prefix}_{s}" for s in suffixes]

    x = np.asarray(data, dtype=float).reshape(-1)
    if x.size == 0:
        return np.zeros(8, dtype=float), names

    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.zeros(8, dtype=float), names

    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    stats = np.array(
        [np.min(x), q1, med, q3, np.max(x), (q3 - q1), np.mean(x), np.std(x, ddof=0)],
        dtype=float,
    )
    return stats, names


# ---------------------------------------------------------------------
# MATLAB: calculate_gammaM.m
# ---------------------------------------------------------------------

def calculate_gammaM(l: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """
    gammaM(i,j) = exp(-||l_i - l_j||^2 / (2*sigma^2))

    l : (TotalPoints, K) locations array
    sigma : if None, uses (2/3) * max side length across dimensions,
            where side length is max(l[:,k]) - min(l[:,k]) + 1.
    """
    l = np.asarray(l, dtype=float)
    if l.ndim != 2:
        raise ValueError("l must be a 2D array (TotalPoints, K).")

    if sigma is None:
        side_lengths = np.max(l, axis=0) - np.min(l, axis=0) + 1.0
        sigma = (2.0 / 3.0) * float(np.max(side_lengths))

    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    dist_mat = squareform(pdist(l, metric="euclidean"))
    return np.exp(-(dist_mat**2) / (2.0 * sigma**2))


# ---------------------------------------------------------------------
# MATLAB: calculate_rhoM.m
# ---------------------------------------------------------------------

def calculate_rhoM(
    S: np.ndarray,
    SF: Optional[float] = None,
    d: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    rm(i,j) = 1 - dist(S_i, S_j) / SF

    d:
      0 -> Euclidean
      1 -> Manhattan (cityblock)
      2 -> Cosine
    """
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if S.ndim != 2:
        raise ValueError("S must be 1D or 2D.")

    metric = {0: "euclidean", 1: "cityblock", 2: "cosine"}.get(d)
    if metric is None:
        raise ValueError("Invalid distance option d. Use 0, 1, or 2.")

    dist_mat = squareform(pdist(S, metric=metric))

    if SF is None:
        SF = float(np.max(dist_mat))
    else:
        SF = float(SF)

    if SF == 0.0:
        SF = 1.0

    rm = 1.0 - (dist_mat / SF)
    return rm, SF


# ---------------------------------------------------------------------
# MATLAB: calculate_tauM.m
# ---------------------------------------------------------------------

def calculate_tauM(rm: np.ndarray, gm: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """
    taum = (rm^(lam/(1-lam))) .* (gm^((1-lam)/lam))
    """
    rm = np.asarray(rm, dtype=float)
    gm = np.asarray(gm, dtype=float)

    if rm.shape != gm.shape:
        raise ValueError(f"Dimension mismatch: rm {rm.shape} vs gm {gm.shape}")

    lam = float(lam)
    if not (0.0 < lam < 1.0):
        raise ValueError("lam (lambda) must be strictly between 0 and 1.")

    exp1 = lam / (1.0 - lam)
    exp2 = (1.0 - lam) / lam
    return (rm**exp1) * (gm**exp2)


# ---------------------------------------------------------------------
# WHRN metrics (new versions using edge 'distance' as cost)
# ---------------------------------------------------------------------

def calculate_global_whrn(
    G: nx.Graph,
    *,
    nodelist: Optional[Sequence[Any]] = None,
    betweenness_normalized: bool = True,
) -> Dict[str, np.ndarray | float]:
    """
    Port of calculate_global_whrn.m (new: uses edge attribute 'distance' as cost).
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    N = len(nodes)
    if N == 0:
        return {
            "weighted_degree": np.array([], dtype=float),
            "edge_density": 0.0,
            "avg_path_length": 0.0,
            "betweenness": np.array([], dtype=float),
            "closeness": np.array([], dtype=float),
            "eigenvector": np.array([], dtype=float),
        }

    if G.number_of_edges() > 0:
        u, v = next(iter(G.edges()))
        if "distance" not in G.edges[u, v]:
            raise ValueError("Edge attribute 'distance' missing. Build with build_graph() or set it.")

    W = adjacency_weight_matrix(G, nodelist=nodes, weight="weight")

    # Eq (12)
    weighted_degree = W.sum(axis=1)

    # Eq (13) undirected density
    edge_density = 0.0 if N < 2 else float(G.number_of_edges()) / (N * (N - 1) / 2.0)

    # Shortest paths using distance-as-cost
    dist_matrix = all_pairs_shortest_path_matrix(G, nodelist=nodes, weight="distance")

    # Eq (14)
    reachable_mask = np.isfinite(dist_matrix) & (dist_matrix > 0)
    if np.any(reachable_mask):
        weighted_distances = dist_matrix[reachable_mask] * W[reachable_mask]
        avg_path_length = float(np.sum(weighted_distances) / np.count_nonzero(reachable_mask))
    else:
        avg_path_length = 0.0

    # Eq (15) betweenness using distance-as-cost
    if G.number_of_edges() == 0:
        betweenness = np.zeros(N, dtype=float)
    else:
        b = nx.betweenness_centrality(G, weight="distance", normalized=betweenness_normalized)
        betweenness = np.array([b.get(node, 0.0) for node in nodes], dtype=float)

    # Eq (16) closeness (match your cross-class definition)
    closeness = np.zeros(N, dtype=float)
    for i in range(N):
        drow = dist_matrix[i, :]
        reachable = drow[np.isfinite(drow) & (drow > 0)]
        if reachable.size > 0:
            closeness[i] = float(reachable.size / np.sum(reachable))

    # Eq (17) eigenvector using strength weight
    if G.number_of_edges() == 0:
        eigenvector = np.zeros(N, dtype=float)
    else:
        try:
            e = nx.eigenvector_centrality_numpy(G, weight="weight")
            eigenvector = np.array([e.get(node, 0.0) for node in nodes], dtype=float)
        except Exception:
            eigenvector = np.zeros(N, dtype=float)

    return {
        "weighted_degree": weighted_degree,
        "edge_density": edge_density,
        "avg_path_length": avg_path_length,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    }


def calculate_inner_class_whrn(
    G: nx.Graph,
    labels: Sequence[int],
    num_classes: int,
    *,
    nodelist: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Port of calculate_inner_class_whrn.m (new).
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    labels = _as_1d_array(labels)
    if labels.size != len(nodes):
        raise ValueError("labels length must match number of nodes (aligned with nodelist).")

    out: List[Dict[str, Any]] = []
    for c in range(1, num_classes + 1):
        idx = np.where(labels == c)[0]
        class_nodes = [nodes[i] for i in idx.tolist()]

        entry = {
            "class_id": c,
            "node_count": len(class_nodes),
            "nodes": class_nodes,
            "metrics": None,
        }

        if len(class_nodes) < 2:
            out.append(entry)
            continue

        SubG = G.subgraph(class_nodes).copy()
        if SubG.number_of_edges() == 0:
            out.append(entry)
            continue

        entry["metrics"] = calculate_global_whrn(SubG, nodelist=class_nodes)
        out.append(entry)

    return out


def calculate_cross_class_whrn(
    G: nx.Graph,
    labels: Sequence[int],
    num_classes: int,
    *,
    nodelist: Optional[Sequence[Any]] = None,
    betweenness_normalized: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Port of calculate_cross_class_whrn.m (new).

    cross_stats[c1-1][c2-1] is a dict with keys:
        density, avg_path, weighted_degree, betweenness, closeness, eigenvector
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    labels = _as_1d_array(labels)
    if labels.size != len(nodes):
        raise ValueError("labels length must match number of nodes (aligned with nodelist).")

    W = adjacency_weight_matrix(G, nodelist=nodes, weight="weight")

    def empty(n: int = 0) -> Dict[str, Any]:
        return {
            "density": 0.0,
            "avg_path": 0.0,
            "weighted_degree": np.zeros(n, dtype=float),
            "betweenness": np.zeros(n, dtype=float),
            "closeness": np.zeros(n, dtype=float),
            "eigenvector": np.zeros(n, dtype=float),
        }

    cross_stats: List[List[Dict[str, Any]]] = [[empty() for _ in range(num_classes)] for _ in range(num_classes)]

    for c1 in range(1, num_classes + 1):
        idx1 = np.where(labels == c1)[0]
        nodes1 = [nodes[i] for i in idx1.tolist()]
        n1 = len(nodes1)

        for c2 in range(c1 + 1, num_classes + 1):
            idx2 = np.where(labels == c2)[0]
            nodes2 = [nodes[i] for i in idx2.tolist()]
            n2 = len(nodes2)

            if n1 == 0 or n2 == 0:
                continue

            W_cross = W[np.ix_(idx1, idx2)]
            deg_c1_to_c2 = W_cross.sum(axis=1)
            deg_c2_to_c1 = W_cross.sum(axis=0)

            num_edges = int(np.count_nonzero(W_cross > 0))
            density_val = float(num_edges) / float(n1 * n2)

            if density_val == 0.0:
                cross_stats[c1 - 1][c2 - 1] = {
                    "density": 0.0,
                    "avg_path": 0.0,
                    "weighted_degree": deg_c1_to_c2,
                    "betweenness": np.zeros(n1, dtype=float),
                    "closeness": np.zeros(n1, dtype=float),
                    "eigenvector": np.zeros(n1, dtype=float),
                }
                cross_stats[c2 - 1][c1 - 1] = {
                    "density": 0.0,
                    "avg_path": 0.0,
                    "weighted_degree": deg_c2_to_c1,
                    "betweenness": np.zeros(n2, dtype=float),
                    "closeness": np.zeros(n2, dtype=float),
                    "eigenvector": np.zeros(n2, dtype=float),
                }
                continue

            combined = nodes1 + nodes2
            SubG = G.subgraph(combined).copy()

            dist_mat = all_pairs_shortest_path_matrix(SubG, nodelist=combined, weight="distance")
            d_c1c2 = dist_mat[0:n1, n1:n1 + n2]

            valid = np.isfinite(d_c1c2) & (d_c1c2 > 0)
            if np.any(valid):
                avg_path_val = float(np.sum(d_c1c2[valid] * W_cross[valid]) / np.count_nonzero(valid))
            else:
                avg_path_val = 0.0

            if SubG.number_of_edges() == 0:
                bet = np.zeros(n1 + n2, dtype=float)
            else:
                b = nx.betweenness_centrality(SubG, weight="distance", normalized=betweenness_normalized)
                bet = np.array([b.get(node, 0.0) for node in combined], dtype=float)

            closeness_c1 = np.zeros(n1, dtype=float)
            for i in range(n1):
                row = d_c1c2[i, :]
                reachable = row[np.isfinite(row) & (row > 0)]
                if reachable.size > 0:
                    closeness_c1[i] = float(reachable.size / np.sum(reachable))

            closeness_c2 = np.zeros(n2, dtype=float)
            for j in range(n2):
                col = d_c1c2[:, j]
                reachable = col[np.isfinite(col) & (col > 0)]
                if reachable.size > 0:
                    closeness_c2[j] = float(reachable.size / np.sum(reachable))

            if SubG.number_of_edges() == 0:
                eig = np.zeros(n1 + n2, dtype=float)
            else:
                try:
                    e = nx.eigenvector_centrality_numpy(SubG, weight="weight")
                    eig = np.array([e.get(node, 0.0) for node in combined], dtype=float)
                except Exception:
                    eig = np.zeros(n1 + n2, dtype=float)

            cross_stats[c1 - 1][c2 - 1] = {
                "density": density_val,
                "avg_path": avg_path_val,
                "weighted_degree": deg_c1_to_c2,
                "betweenness": bet[0:n1],
                "closeness": closeness_c1,
                "eigenvector": eig[0:n1],
            }
            cross_stats[c2 - 1][c1 - 1] = {
                "density": density_val,
                "avg_path": avg_path_val,
                "weighted_degree": deg_c2_to_c1,
                "betweenness": bet[n1:n1 + n2],
                "closeness": closeness_c2,
                "eigenvector": eig[n1:n1 + n2],
            }

    return cross_stats


# ---------------------------------------------------------------------
# Old versions (ported from *_old.m)
# ---------------------------------------------------------------------

def calculate_global_whrn_old(
    G: nx.Graph,
    W: np.ndarray,
    *,
    nodelist: Optional[Sequence[Any]] = None,
    betweenness_normalized: bool = True,
) -> Dict[str, np.ndarray | float]:
    """
    Port of calculate_global_whrn_old.m:
      - shortest paths use edge attribute 'weight' as COST
      - edge density denominator matches old code: N*(N-1)
      - W is passed explicitly
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] != len(nodes):
        raise ValueError("W must be NxN and aligned with nodelist.")

    N = len(nodes)

    weighted_degree = W.sum(axis=1)
    edge_density = 0.0 if N < 2 else float(G.number_of_edges()) / float(N * (N - 1))

    dist_matrix = all_pairs_shortest_path_matrix(G, nodelist=nodes, weight="weight")

    reachable_mask = np.isfinite(dist_matrix) & (dist_matrix > 0)
    if np.any(reachable_mask):
        weighted_distances = dist_matrix[reachable_mask] * W[reachable_mask]
        avg_path_length = float(np.sum(weighted_distances) / np.count_nonzero(reachable_mask))
    else:
        avg_path_length = 0.0

    if G.number_of_edges() == 0:
        betweenness = np.zeros(N, dtype=float)
        eigenvector = np.zeros(N, dtype=float)
    else:
        b = nx.betweenness_centrality(G, weight="weight", normalized=betweenness_normalized)
        betweenness = np.array([b.get(node, 0.0) for node in nodes], dtype=float)

        try:
            e = nx.eigenvector_centrality_numpy(G, weight="weight")
            eigenvector = np.array([e.get(node, 0.0) for node in nodes], dtype=float)
        except Exception:
            eigenvector = np.zeros(N, dtype=float)

    closeness = np.zeros(N, dtype=float)
    for i in range(N):
        drow = dist_matrix[i, :]
        reachable = drow[np.isfinite(drow) & (drow > 0)]
        if reachable.size > 0:
            closeness[i] = float(reachable.size / np.sum(reachable))

    return {
        "weighted_degree": weighted_degree,
        "edge_density": edge_density,
        "avg_path_length": avg_path_length,
        "betweenness": betweenness,
        "closeness": closeness,
        "eigenvector": eigenvector,
    }


def calculate_inner_class_whrn_old(
    G: nx.Graph,
    W: np.ndarray,
    labels: Sequence[int],
    num_classes: int,
    *,
    nodelist: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Port of calculate_inner_class_whrn_old.m
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    labels = _as_1d_array(labels)
    if labels.size != len(nodes):
        raise ValueError("labels length must match number of nodes (aligned with nodelist).")

    W = np.asarray(W, dtype=float)
    if W.shape != (len(nodes), len(nodes)):
        raise ValueError("W must be NxN and aligned with nodelist.")

    out: List[Dict[str, Any]] = []
    for c in range(1, num_classes + 1):
        idx = np.where(labels == c)[0]
        class_nodes = [nodes[i] for i in idx.tolist()]

        if idx.size < 2:
            out.append({"class_id": c, "node_count": int(idx.size), "nodes": class_nodes, "metrics": None})
            continue

        SubG = G.subgraph(class_nodes).copy()
        SubW = W[np.ix_(idx, idx)]
        metrics = calculate_global_whrn_old(SubG, SubW, nodelist=class_nodes)

        out.append({"class_id": c, "node_count": len(class_nodes), "nodes": class_nodes, "metrics": metrics})

    return out


def calculate_cross_class_whrn_old(
    G: nx.Graph,
    W: np.ndarray,
    labels: Sequence[int],
    num_classes: int,
    *,
    nodelist: Optional[Sequence[Any]] = None,
    betweenness_normalized: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Port of calculate_cross_class_whrn_old.m
    """
    if nodelist is None:
        nodes = _default_nodelist(G)
    else:
        nodes = list(nodelist)

    labels = _as_1d_array(labels)
    if labels.size != len(nodes):
        raise ValueError("labels length must match number of nodes (aligned with nodelist).")

    W = np.asarray(W, dtype=float)
    if W.shape != (len(nodes), len(nodes)):
        raise ValueError("W must be NxN and aligned with nodelist.")

    def empty(n: int = 0) -> Dict[str, Any]:
        return {
            "density": 0.0,
            "avg_path": 0.0,
            "weighted_degree": np.zeros(n, dtype=float),
            "betweenness": np.zeros(n, dtype=float),
            "closeness": np.zeros(n, dtype=float),
            "eigenvector": np.zeros(n, dtype=float),
        }

    cross_stats: List[List[Dict[str, Any]]] = [[empty() for _ in range(num_classes)] for _ in range(num_classes)]

    for c1 in range(1, num_classes + 1):
        idx1 = np.where(labels == c1)[0]
        nodes1 = [nodes[i] for i in idx1.tolist()]
        n1 = len(nodes1)

        for c2 in range(c1 + 1, num_classes + 1):
            idx2 = np.where(labels == c2)[0]
            nodes2 = [nodes[i] for i in idx2.tolist()]
            n2 = len(nodes2)

            if n1 == 0 or n2 == 0:
                continue

            W_cross = W[np.ix_(idx1, idx2)]
            deg_c1_to_c2 = W_cross.sum(axis=1)
            deg_c2_to_c1 = W_cross.sum(axis=0)

            num_edges = int(np.count_nonzero(W_cross > 0))
            density_val = float(num_edges) / float(n1 * n2)

            if density_val == 0.0:
                cross_stats[c1 - 1][c2 - 1] = {
                    "density": 0.0,
                    "avg_path": 0.0,
                    "weighted_degree": deg_c1_to_c2,
                    "betweenness": np.zeros(n1, dtype=float),
                    "closeness": np.zeros(n1, dtype=float),
                    "eigenvector": np.zeros(n1, dtype=float),
                }
                cross_stats[c2 - 1][c1 - 1] = {
                    "density": 0.0,
                    "avg_path": 0.0,
                    "weighted_degree": deg_c2_to_c1,
                    "betweenness": np.zeros(n2, dtype=float),
                    "closeness": np.zeros(n2, dtype=float),
                    "eigenvector": np.zeros(n2, dtype=float),
                }
                continue

            combined = nodes1 + nodes2
            SubG = G.subgraph(combined).copy()

            # OLD: use 'weight' as cost
            dist_mat = all_pairs_shortest_path_matrix(SubG, nodelist=combined, weight="weight")
            d_c1c2 = dist_mat[0:n1, n1:n1 + n2]

            valid = np.isfinite(d_c1c2) & (d_c1c2 > 0)
            if np.any(valid):
                avg_path_val = float(np.sum(d_c1c2[valid] * W_cross[valid]) / np.count_nonzero(valid))
            else:
                avg_path_val = 0.0

            if SubG.number_of_edges() == 0:
                bet = np.zeros(n1 + n2, dtype=float)
                eig = np.zeros(n1 + n2, dtype=float)
            else:
                b = nx.betweenness_centrality(SubG, weight="weight", normalized=betweenness_normalized)
                bet = np.array([b.get(node, 0.0) for node in combined], dtype=float)

                try:
                    e = nx.eigenvector_centrality_numpy(SubG, weight="weight")
                    eig = np.array([e.get(node, 0.0) for node in combined], dtype=float)
                except Exception:
                    eig = np.zeros(n1 + n2, dtype=float)

            # Cross closeness (same logic)
            closeness_c1 = np.zeros(n1, dtype=float)
            for i in range(n1):
                row = d_c1c2[i, :]
                reachable = row[np.isfinite(row) & (row > 0)]
                if reachable.size > 0:
                    closeness_c1[i] = float(reachable.size / np.sum(reachable))

            closeness_c2 = np.zeros(n2, dtype=float)
            for j in range(n2):
                col = d_c1c2[:, j]
                reachable = col[np.isfinite(col) & (col > 0)]
                if reachable.size > 0:
                    closeness_c2[j] = float(reachable.size / np.sum(reachable))

            cross_stats[c1 - 1][c2 - 1] = {
                "density": density_val,
                "avg_path": avg_path_val,
                "weighted_degree": deg_c1_to_c2,
                "betweenness": bet[0:n1],
                "closeness": closeness_c1,
                "eigenvector": eig[0:n1],
            }
            cross_stats[c2 - 1][c1 - 1] = {
                "density": density_val,
                "avg_path": avg_path_val,
                "weighted_degree": deg_c2_to_c1,
                "betweenness": bet[n1:n1 + n2],
                "closeness": closeness_c2,
                "eigenvector": eig[n1:n1 + n2],
            }

    return cross_stats
