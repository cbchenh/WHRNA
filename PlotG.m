% Plot
figure;
p = plot(G, 'Layout', 'force');

% Node colors by property
p.NodeCData = G.Nodes.Property;
colormap(jet);  % or parula, hot, cool, etc.
c = colorbar;
c.Label.String = 'Node Property';

% Edge thickness by weight
edge_weights = G.Edges.Weight;
min_width = 0.5;
max_width = 4;
p.LineWidth = min_width + (max_width - min_width) * ...
              normalize(edge_weights, 'range');

% Optional: Edge color by weight too
p.EdgeCData = edge_weights;
p.EdgeAlpha = 0.55;  % Transparency

% Node size (optional)
p.MarkerSize = 2;

title('Weighted Network');
axis off;