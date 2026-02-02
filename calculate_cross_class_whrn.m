function cross_stats = calculate_cross_class_whrn(G, labels, num_classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE_CROSS_CLASS_WHRN
% Cross-class WHRN metrics between cluster pairs.
% Uses G.Edges.Distance as edge cost for shortest paths / betweenness.
%
% Inputs:
%   G           : graph object with G.Edges.Distance and G.Edges.Weight
%   labels      : Nx1 class labels
%   num_classes : number of classes
%
% Output:
%   cross_stats : num_classes x num_classes struct
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    N = numnodes(G);
    if numel(labels) ~= N
        error('labels must have length N = numnodes(G).');
    end
    labels = labels(:);

    if ~ismember('Distance', G.Edges.Properties.VariableNames)
        error('G.Edges.Distance is missing. Build G with buildG() so Distance is defined.');
    end

    % Weight adjacency from G (source of truth)
    W = adjacency(G, 'weighted');

    % Initialize struct matrix
    cross_stats = repmat(struct( ...
        'density', [], 'avg_path', [], 'weighted_degree', [], ...
        'betweenness', [], 'closeness', [], 'eigenvector', []), ...
        num_classes, num_classes);

    for c1 = 1:num_classes
        idx1 = find(labels == c1);
        n1 = numel(idx1);

        for c2 = (c1 + 1):num_classes
            idx2 = find(labels == c2);
            n2 = numel(idx2);

            if isempty(idx1) || isempty(idx2)
                continue;
            end

            % --- Cross weighted degree + density based on cross block of W ---
            W_cross = W(idx1, idx2);
            deg_c1_to_c2 = sum(W_cross, 2);
            deg_c2_to_c1 = sum(W_cross, 1)';

            num_edges = nnz(W_cross > 0);
            density_val = num_edges / (n1 * n2);

            % If no interaction edges, set zeros and skip expensive computations
            if density_val == 0
                cross_stats(c1,c2).density = 0;
                cross_stats(c1,c2).avg_path = 0;
                cross_stats(c1,c2).weighted_degree = deg_c1_to_c2;
                cross_stats(c1,c2).betweenness = zeros(n1, 1);
                cross_stats(c1,c2).closeness = zeros(n1, 1);
                cross_stats(c1,c2).eigenvector = zeros(n1, 1);

                cross_stats(c2,c1).density = 0;
                cross_stats(c2,c1).avg_path = 0;
                cross_stats(c2,c1).weighted_degree = deg_c2_to_c1;
                cross_stats(c2,c1).betweenness = zeros(n2, 1);
                cross_stats(c2,c1).closeness = zeros(n2, 1);
                cross_stats(c2,c1).eigenvector = zeros(n2, 1);

                continue;
            end

            % --- Build mesoscale subgraph containing both classes ---
            combined_idx = [idx1; idx2];
            SubG = subgraph(G, combined_idx);

            % Local index mapping inside SubG
            loc1 = 1:n1;
            loc2 = (n1+1):(n1+n2);

            % Prepare cost-graph for older MATLAB distances()/centrality()
            cost = SubG.Edges.Distance;

            SubGcost = SubG;
            SubGcost.Edges.Weight = cost;   % <-- critical: distances uses Weight as cost

            % --- Shortest paths using COST (Distance) ---
            dist_mat = distances(SubGcost, 'Method', 'positive');

            % Distances from class1 nodes to class2 nodes
            d_c1c2 = dist_mat(loc1, loc2);

            % --- Cross Avg Path Length (weighted by W_cross like your earlier code) ---
            valid_paths = isfinite(d_c1c2) & (d_c1c2 > 0);
            if any(valid_paths(:))
                avg_path_val = sum(d_c1c2(valid_paths) .* W_cross(valid_paths)) / nnz(valid_paths);
            else
                avg_path_val = 0;
            end

            % --- Cross Betweenness Centrality (use cost) ---
            % Prefer Cost argument if available; otherwise fall back to SubGcost
            try
                bet_vec = centrality(SubG, 'betweenness', 'Cost', cost);
            catch
                bet_vec = centrality(SubGcost, 'betweenness');
            end

            % --- Cross Closeness Centrality (custom: only across clusters) ---
            closeness_c1 = zeros(n1, 1);
            for i = 1:n1
                d_row = d_c1c2(i, :);
                reachable = d_row(isfinite(d_row) & d_row > 0);
                if ~isempty(reachable)
                    closeness_c1(i) = numel(reachable) / sum(reachable);
                end
            end

            closeness_c2 = zeros(n2, 1);
            for j = 1:n2
                d_col = d_c1c2(:, j);
                reachable = d_col(isfinite(d_col) & d_col > 0);
                if ~isempty(reachable)
                    closeness_c2(j) = numel(reachable) / sum(reachable);
                end
            end

            % --- Cross Eigenvector Centrality (use strength, not cost) ---
            % Use SubG.Edges.Weight (original strengths) as importance
            try
                eig_vec = centrality(SubG, 'eigenvector', 'Importance', SubG.Edges.Weight);
            catch
                % fallback: if Importance name-value unsupported, default call
                eig_vec = centrality(SubG, 'eigenvector');
            end

            % --- Store c1 -> c2 ---
            cross_stats(c1,c2).density = density_val;
            cross_stats(c1,c2).avg_path = avg_path_val;
            cross_stats(c1,c2).weighted_degree = deg_c1_to_c2;
            cross_stats(c1,c2).betweenness = bet_vec(loc1);
            cross_stats(c1,c2).closeness = closeness_c1;
            cross_stats(c1,c2).eigenvector = eig_vec(loc1);

            % --- Store c2 -> c1 ---
            cross_stats(c2,c1).density = density_val;
            cross_stats(c2,c1).avg_path = avg_path_val;
            cross_stats(c2,c1).weighted_degree = deg_c2_to_c1;
            cross_stats(c2,c1).betweenness = bet_vec(loc2);
            cross_stats(c2,c1).closeness = closeness_c2;
            cross_stats(c2,c1).eigenvector = eig_vec(loc2);
        end
    end
end
