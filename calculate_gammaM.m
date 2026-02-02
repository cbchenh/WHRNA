function gM = calculate_gammaM(l, sigma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gammaM(i,j) = Gaussian(||i - j||)/Gaussian(||0||)                       %
% - Input:                                                                %
%   l: locations matrix (TotalPoints x K) from flatten_to_sequence        %
%   sigma: std of Gaussian. If empty, uses 2/3 of the max side length.    %
% - Output:                                                               %
%   gM: standardized closeness                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [total_points, dims] = size(l);
    
    % 1. Handle sigma calculation if empty
    if nargin < 2 || isempty(sigma)
        % For generalized shapes, we find the range of each dimension
        % range = max(coordinate) - min(coordinate) + 1
        side_lengths = max(l, [], 1) - min(l, [], 1) + 1;
        N_max = max(side_lengths); 
        sigma = (2/3) * N_max;
    end
    
    % 2. Calculate Pairwise Euclidean Distances
    dist_mat = squareform(pdist(l, 'euclidean'));
    
    % 3. Apply Standardized Gaussian Kernel
    % G(0) is 1, so we just compute the exponent
    gM = exp(-(dist_mat.^2) / (2 * sigma^2));
end