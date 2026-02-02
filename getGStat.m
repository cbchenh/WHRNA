function [feat_row, header] = getGStat(G, labels, num_classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GETGSTAT (Detailed Version)
% Computes WHRN metrics and organizes them by specific State (Inner)
% and specific Pair (Cross).
%
% Inputs:
%   G           : Global MATLAB graph object (must include G.Edges.Distance)
%   labels      : Nx1 vector of class labels
%   num_classes : Total number of classes (C)
%
% Outputs:
%   feat_row    : 1xM vector of extracted features
%   header      : 1xM cell array of feature names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% 0. Validation
    N = numnodes(G);
    if numel(labels) ~= N
        error('labels must have length N = numnodes(G).');
    end
    labels = labels(:);

    if ~ismember('Distance', G.Edges.Properties.VariableNames)
        error('G.Edges.Distance is missing. Build G with buildG() so Distance is defined.');
    end

    %% 1. Execute Calculations (NEW signatures; no W)
    global_stats = calculate_global_whrn(G);
    inner_stats  = calculate_inner_class_whrn(G, labels, num_classes);
    cross_stats  = calculate_cross_class_whrn(G, labels, num_classes);

    %% 2. Initialize
    feat_row = [];
    header = {};

    %% 3. Global Statistics (One set for the whole graph)
    % Scalars
    feat_row = [feat_row, global_stats.edge_density, global_stats.avg_path_length];
    header   = [header, {'Global_Density', 'Global_AvgPath'}];

    % Distributions
    [stats, names] = calc_dist_stats(global_stats.weighted_degree, 'Global_Degree');
    feat_row = [feat_row, stats]; header = [header, names];

    [stats, names] = calc_dist_stats(global_stats.betweenness, 'Global_Betweenness');
    feat_row = [feat_row, stats]; header = [header, names];

    [stats, names] = calc_dist_stats(global_stats.closeness, 'Global_Closeness');
    feat_row = [feat_row, stats]; header = [header, names];

    [stats, names] = calc_dist_stats(global_stats.eigenvector, 'Global_Eigenvector');
    feat_row = [feat_row, stats]; header = [header, names];

    %% 4. Inner-Class Statistics (Iterate per Class)
    for c = 1:num_classes
        c_prefix = sprintf('Inner_C%d', c);

        if c <= length(inner_stats) && ~isempty(inner_stats(c).metrics)
            m = inner_stats(c).metrics;

            % Scalar Features
            feat_row = [feat_row, m.edge_density, m.avg_path_length];
            header   = [header, {[c_prefix, '_Density'], [c_prefix, '_AvgPath']}];

            % Distribution Features
            [stats, names] = calc_dist_stats(m.weighted_degree, [c_prefix, '_Degree']);
            feat_row = [feat_row, stats]; header = [header, names];

            [stats, names] = calc_dist_stats(m.betweenness, [c_prefix, '_Betweenness']);
            feat_row = [feat_row, stats]; header = [header, names];

            [stats, names] = calc_dist_stats(m.closeness, [c_prefix, '_Closeness']);
            feat_row = [feat_row, stats]; header = [header, names];

            [stats, names] = calc_dist_stats(m.eigenvector, [c_prefix, '_Eigenvector']);
            feat_row = [feat_row, stats]; header = [header, names];
        else
            % 2 scalars + 4 distributions * 8 stats = 34 zeros
            feat_row = [feat_row, zeros(1, 34)];

            header = [header, {[c_prefix, '_Density'], [c_prefix, '_AvgPath']}];
            [~, n1] = calc_dist_stats([], [c_prefix, '_Degree']);       header = [header, n1];
            [~, n2] = calc_dist_stats([], [c_prefix, '_Betweenness']);  header = [header, n2];
            [~, n3] = calc_dist_stats([], [c_prefix, '_Closeness']);    header = [header, n3];
            [~, n4] = calc_dist_stats([], [c_prefix, '_Eigenvector']);  header = [header, n4];
        end
    end

    %% 5. Cross-Class Statistics (Iterate per Pair)
    for c1 = 1:num_classes
        for c2 = (c1+1):num_classes
            pair_prefix = sprintf('Cross_C%d_C%d', c1, c2);

            s     = cross_stats(c1, c2);
            s_sym = cross_stats(c2, c1);

            % density should always exist (0 or >0) if calculate_cross_class_whrn is correct
            feat_row = [feat_row, s.density, s.avg_path];
            header   = [header, {[pair_prefix, '_Density'], [pair_prefix, '_AvgPath']}];

            % Pool both directions for interaction profile
            deg_pool = [s.weighted_degree; s_sym.weighted_degree];
            [stats, names] = calc_dist_stats(deg_pool, [pair_prefix, '_Degree']);
            feat_row = [feat_row, stats]; header = [header, names];

            bet_pool = [s.betweenness; s_sym.betweenness];
            [stats, names] = calc_dist_stats(bet_pool, [pair_prefix, '_Betweenness']);
            feat_row = [feat_row, stats]; header = [header, names];

            clo_pool = [s.closeness; s_sym.closeness];
            [stats, names] = calc_dist_stats(clo_pool, [pair_prefix, '_Closeness']);
            feat_row = [feat_row, stats]; header = [header, names];

            eig_pool = [s.eigenvector; s_sym.eigenvector];
            [stats, names] = calc_dist_stats(eig_pool, [pair_prefix, '_Eigenvector']);
            feat_row = [feat_row, stats]; header = [header, names];
        end
    end
end
