function [E, W] = tau_to_graph(tau, epsilon, method, range_min)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert similarity tau to Adjacency Matrix E and Weight Matrix W        %
% - Input:                                                                %
%   tau: NxN similarity matrix (0 <= tau <= 1)                            %
%   epsilon: threshold for edge existence                                 %
%   method: 1: Linear, 2: Logarithmic, 3: Soft-Thresholding               %
%   range_min: Default 0.2 for Linear scaling                             %
% - Output:                                                               %
%   E: Adjacency matrix (0 or 1)                                          %
%   W: Recurrence Weight matrix                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if nargin < 3 || isempty(method), method = 1; end
    if nargin < 4 || isempty(range_min), range_min = 0.2; end

    % 1. Adjacency Matrix E based on H(epsilon - |1 - tau|)
    % This extracts values in the range [1-epsilon, 1]
    E = (1 - tau) <= epsilon;
    
    % Optional: Remove self-loops
    E = E - diag(diag(E));
    E(E < 0) = 0; 

    % Initialize Weight Matrix
    W = zeros(size(tau));
    if ~any(E(:)), return; end % Return zeros if no edges meet threshold

    % Extract valid tau values
    tau_active = tau(E == 1);
    lower_bound = 1 - epsilon;

    % 2. Scaling Methods
    switch method
        case 1 % Linear Min-Max Rescaling (Mapped to [range_min, 1])
            % Normalize active tau to [0, 1]
            tau_norm = (tau_active - lower_bound) / epsilon;
            % Scale to [range_min, 1]
            W(E == 1) = range_min + (1 - range_min) * tau_norm;

        case 2 % Logarithmic Scaling
            % Expands the differences near 1
            % Maps 1-epsilon to ~0.36 and 1 to 1.0
            W(E == 1) = exp(-(1 - tau_active) / epsilon);

        case 3 % Soft-Thresholding (Sigmoid)
            % Center the sigmoid at the midpoint of the epsilon window
            midpoint = 1 - (epsilon / 2);
            k = 10 / epsilon; % Steepness factor
            W(E == 1) = 1 ./ (1 + exp(-k * (tau_active - midpoint)));

        otherwise
            error('Invalid method option. Use 1, 2, or 3.');
    end
    
    % Ensure weights don't exceed 1 due to floating point precision
    W(W > 1) = 1;
end