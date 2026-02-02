function stats = calculate_global_whrn(G)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE_GLOBAL_WHRN
% Uses G.Edges.Distance as edge cost for shortest paths / betweenness / closeness.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    N = numnodes(G);

    % Ensure Distance exists
    if ~ismember('Distance', G.Edges.Properties.VariableNames)
        error('G.Edges.Distance is missing. Build G with buildG() first.');
    end

    % Weight adjacency (for weighted degree and any w_ij usage)
    W = adjacency(G, 'weighted');

    % --- Eq. (12): Global Weighted Degree (D_i) ---
    stats.weighted_degree = sum(W, 2);

    % --- Eq. (13): Global Edge Density (pi) ---
    stats.edge_density = numedges(G) / (N * (N - 1) / 2);

    % ---- Use edge DISTANCE as COST for shortest paths and related metrics ----
    cost = G.Edges.Distance;

    % Create a temporary graph whose "Weight" is the COST
    Gcost = G;
    Gcost.Edges.Weight = cost;

    % --- Shortest Path Matrix for Equations (14), (15), (16) ---
    % Older MATLAB: distances() uses G.Edges.Weight automatically; no 'Weights' arg.
    dist_matrix = distances(Gcost, 'Method', 'positive');

    % --- Eq. (14): Global Average Path Length (L) ---
    reachable_mask = isfinite(dist_matrix) & (dist_matrix > 0);

    if any(reachable_mask(:))
        weighted_distances = dist_matrix(reachable_mask) .* W(reachable_mask);
        stats.avg_path_length = sum(weighted_distances) / nnz(reachable_mask);
    else
        stats.avg_path_length = 0;
    end

    % --- Eq. (15): Global Betweenness Centrality (b_p) ---
    % Try 'Cost' (newer versions). If unsupported, fall back to using Gcost weights.
    try
        stats.betweenness = centrality(G, 'betweenness', 'Cost', cost);
    catch
        stats.betweenness = centrality(Gcost, 'betweenness');
    end

    % --- Eq. (16): Global Closeness Centrality (C_i) ---
    try
        stats.closeness = centrality(G, 'closeness', 'Cost', cost);
    catch
        stats.closeness = centrality(Gcost, 'closeness');
    end

    % --- Eq. (17): Global Eigenvector Centrality (E_i) ---
    % Use original edge strengths (NOT distance) for eigenvector "importance"
    stats.eigenvector = centrality(G, 'eigenvector', 'Importance', G.Edges.Weight);
end
