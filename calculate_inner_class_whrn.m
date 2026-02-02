function inner_stats = calculate_inner_class_whrn(G, labels, num_classes)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALCULATE_INNER_CLASS_WHRN
% Extracts subgraphs for each class and calculates metrics for each sub-WHRN.
%
% Inputs:
%   G           : Global MATLAB graph object (must include G.Edges.Distance)
%   labels      : Nx1 vector of class labels
%   num_classes : Scalar, total number of classes (C)
%
% Output:
%   inner_stats : 1 x num_classes struct array containing metrics for each
%                 sub-WHRN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Preallocate output struct array
    inner_stats(1, num_classes) = struct( ...
        'class_id', [], ...
        'node_count', [], ...
        'node_indices', [], ...
        'metrics', [] ...
    );

    % Basic validation
    N = numnodes(G);
    if numel(labels) ~= N
        error('labels must have length N = numnodes(G).');
    end
    labels = labels(:);

    % Ensure Distance exists (required for shortest-path based metrics)
    if ~ismember('Distance', G.Edges.Properties.VariableNames)
        error('G.Edges.Distance is missing. Build G with buildG() so Distance is defined.');
    end

    for c = 1:num_classes
        class_indices = find(labels == c);

        inner_stats(c).class_id = c;
        inner_stats(c).node_count = numel(class_indices);
        inner_stats(c).node_indices = class_indices;

        % Skip tiny classes (no edges possible)
        if numel(class_indices) < 2
            inner_stats(c).metrics = [];
            continue;
        end

        % Subgraph retains edge tables, including Distance and Weight
        SubG = subgraph(G, class_indices);

        % If SubG ends up with 0 edges, metrics that depend on paths may be trivial
        if numedges(SubG) == 0
            inner_stats(c).metrics = [];
            continue;
        end

        % Use revised global function that uses SubG.Edges.Distance internally
        inner_stats(c).metrics = calculate_global_whrn(SubG);
    end
end
