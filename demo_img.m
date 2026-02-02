tic
%% 1. Setup Data
img = imread('demo.png');

% Ensure feature dimension D = number of channels
if ndims(img) == 2
    D = 1;                 % grayscale
elseif ndims(img) == 3
    D = size(img, 3);      % RGB or multi-channel
else
    error('Unsupported image dimensions.');
end

% Convert to double for downstream computations (kmeans, similarity, etc.)
imgD = double(img);

toc

%% 2. Process through your functions
% Step A: Flatten and get coordinates
[S_seq, l] = flatten_to_sequence(imgD, D);

% Step B: Calculate State Similarity (rhoM)
[rm, SF] = calculate_rhoM(S_seq, [], 0);

% Step C: Calculate Spatial Closeness (gammaM)
gm = calculate_gammaM(l, []);

% Step D: Combine into tauM
lambda = 0.5;
taum = calculate_tauM(rm, gm, lambda);

% Step E: Generate Adjacency and Weight Matrices
epsilon = 0.085;
method = 1;
[E, W] = tau_to_graph(taum, epsilon, method);

% Step F: Create node property by clustering into K clusters
K = 16;
Nidx = kmeans(S_seq, K);

toc

%% 3. Visualization (optional; can be heavy for large images)
figure('Position', [100, 100, 1200, 400]);

subplot(1,4,1);
imagesc(img);
axis square; axis off;
title('Original Data');

subplot(1,4,2);
imagesc(rm);
axis square; colorbar;
title('rhoM (Value Similarity)');

subplot(1,4,3);
imagesc(gm);
axis square; colorbar;
title('gammaM (Spatial Closeness)');

subplot(1,4,4);
imagesc(taum);
axis square; colorbar;
title('tauM (Combined)');

% Optional graph layout in pixel coordinates (can be huge)
% (Use only for small images; otherwise it will be slow / memory-heavy)
%{
figure;
Gbin = graph(E);  % binary adjacency
p = plot(Gbin, 'XData', l(:,2), 'YData', -l(:,1), 'NodeLabel', {});
p.EdgeAlpha = 0.3;
title('Binary Graph Visualization (E)');
axis tight; axis off;
%}

%% 4. Build Weighted Recurrence Network
tic
G = buildG(W, Nidx);   % buildG must create G.Edges.Distance
toc

% Quick sanity check
assert(ismember('Distance', G.Edges.Properties.VariableNames), 'G.Edges.Distance missing.');

%% 5. Plot weighted network (your script)
tic
PlotG
toc

%% 6. WHRN Quantifications (NEW signatures; no W)
tic
global_stats = calculate_global_whrn(G);
toc

tic
inner_stats  = calculate_inner_class_whrn(G, Nidx, K);
toc

tic
cross_stats  = calculate_cross_class_whrn(G, Nidx, K);
toc

tic
[feat_row, header] = getGStat(G, Nidx, K);
toc
