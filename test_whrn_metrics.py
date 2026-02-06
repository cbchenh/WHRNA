"""Quick sanity tests for whrn_metrics.py

Run:
    python test_whrn_metrics.py
"""

import os, sys
import numpy as np

# Ensure this folder is on the Python path
sys.path.insert(0, os.path.dirname(__file__))

from whrn_metrics import (
    build_graph,
    calculate_global_whrn,
    calculate_inner_class_whrn,
    calculate_cross_class_whrn,
    calculate_gammaM,
    calculate_rhoM,
    calculate_tauM,
    calc_dist_stats,
    all_pairs_shortest_path_matrix,
)

def main():
    W = np.array([
        [0, 2, 0, 0],
        [2, 0, 1, 0],
        [0, 1, 0, 3],
        [0, 0, 3, 0],
    ], dtype=float)

    props = np.arange(1, 5)
    G = build_graph(W, props, epsilon=1e-12, assume_symmetric=True)

    dist = all_pairs_shortest_path_matrix(G, nodelist=[1,2,3,4], weight="distance")
    expected_1_to_4 = (1/2) + (1/1) + (1/3)
    assert abs(dist[0,3] - expected_1_to_4) < 1e-9, (dist[0,3], expected_1_to_4)

    stats = calculate_global_whrn(G, nodelist=[1,2,3,4])
    print("Global edge_density:", stats["edge_density"])
    print("Global weighted_degree:", stats["weighted_degree"])
    print("Global avg_path_length:", stats["avg_path_length"])

    assert abs(stats["edge_density"] - 0.5) < 1e-12
    assert np.allclose(stats["weighted_degree"], np.array([2, 3, 4, 3], dtype=float))

    labels = np.array([1, 1, 2, 2])
    inner = calculate_inner_class_whrn(G, labels, num_classes=2, nodelist=[1,2,3,4])
    print("\nInner-class node_counts:", [x["node_count"] for x in inner])

    cross = calculate_cross_class_whrn(G, labels, num_classes=2, nodelist=[1,2,3,4])
    print("\nCross density (1->2):", cross[0][1]["density"])
    assert abs(cross[0][1]["density"] - 0.25) < 1e-12

    S = np.array([0, 1, 3, 6], dtype=float)
    rm, SF = calculate_rhoM(S, SF=None, d=0)
    l = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    gm = calculate_gammaM(l, sigma=None)
    taum = calculate_tauM(rm, gm, lam=0.5)
    assert rm.shape == gm.shape == taum.shape

    deg_stats, deg_names = calc_dist_stats(stats["weighted_degree"], prefix="Global_Degree")
    print("\nDist stat names:", deg_names)
    print("Dist stats:", deg_stats)

    print("\nAll sanity tests passed.")

if __name__ == "__main__":
    main()
