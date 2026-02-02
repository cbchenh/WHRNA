function G = buildG(W, node_property, epsilon)
% buildG  Build an undirected weighted graph from adjacency matrix W
%         and attach:
%           - G.Nodes.Property
%           - G.Edges.Distance = 1./(G.Edges.Weight + epsilon)

    if nargin < 3 || isempty(epsilon)
        epsilon = 1e-12;
    end

    if ~isnumeric(W) || ~ismatrix(W) || size(W,1) ~= size(W,2)
        error('W must be a numeric NxN matrix.');
    end
    N = size(W,1);

    if size(node_property,1) ~= N
        error('node_property must have N rows (same as size(W,1)).');
    end

    % Build undirected graph from symmetric W
    G = graph(W, 'upper');

    % Node property
    G.Nodes.Property = node_property;

    % Edge distance derived from edge weights
    G.Edges.Distance = 1 ./ (G.Edges.Weight + epsilon);
end