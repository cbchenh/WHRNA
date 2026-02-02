tic
%% 1. Setup Data: Lorenz System Simulation 
% Parameters for chaos: sigma=10, beta=8/3, rho=28
sigma = 10; beta = 8/3; rho = 28;
lorenz_ode = @(t,a) [-sigma*a(1) + sigma*a(2); ...
                     rho*a(1) - a(2) - a(1)*a(3); ...
                     -beta*a(3) + a(1)*a(2)];

% Simulate trajectory
[t, S_raw] = ode45(lorenz_ode, [0 150], [1 1 1]);

% Downsample for network analysis (TotalPoints x Features)
data = S_raw(1:10:end, :); 
D = size(data, 2); % Feature dimension (3 for Lorenz)
toc

%% 2. Process through WHRN functions 
% Step A: Flatten to sequence and get temporal coordinates
[S_seq, l] = flatten_to_sequence(data, D);

% Step B: Calculate State Similarity (rhoM) 
[rm, SF] = calculate_rhoM(S_seq, [], 0); 

% Step C: Calculate Temporal Closeness (gammaM) 
gm = calculate_gammaM(l, []); 

% Step D: Combine into Recurrence Strength (tauM) 
lambda = 0.5;
taum = calculate_tauM(rm, gm, lambda);

% Step E: Generate Adjacency and Weight Matrices 
epsilon = 0.15; 
method = 1; % Linear scaling
[E, W] = tau_to_graph(taum, epsilon, method);

% Step F: Identify Heterogeneous Node Properties (Clustering) 
K = 16; % Number of state-space regions
Nidx = kmeans(S_seq, K); 

toc
tic
% Build Graph 
G = buildG(W, Nidx);
G = buildG(W, Nidx);
toc
tic
%% 3. Visualization
% Plot Graph
PlotG; % Using your script to visualize Weighted Network
title('Lorenz WHRN: Nodes colored by state-space region');

%% 4. WHRN Quantifications (Equations 12-23) [cite: 272, 391]
tic
% (1) Global Statistics (Eq. 12-17)
global_stats = calculate_global_whrn(G);

% (2) Inner-Class Statistics (Eq. 323-326)
inner_stats = calculate_inner_class_whrn(G, Nidx, K);

% (3) Cross-Class Statistics (Eq. 18-23)
cross_stats = calculate_cross_class_whrn(G, Nidx, K);

% (4) Feature Vector Extraction for Machine Learning [cite: 394, 417]
[feat_row, header] = getGStat(G, Nidx, K);
toc