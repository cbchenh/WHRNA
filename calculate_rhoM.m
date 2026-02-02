function [rm, SF] = calculate_rhoM(S, SF, d)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rhoM(i,j) = 1 - ||S_i - S_j||/Scaled Factor                             %
% - Input:                                                                %
%   S: Sequence of states {S_t}:                                          %
%                  by default, the data is 1d N length timeseries         %
%                  if data is a 2d image, convert it into a N^2*D sequence%
%                  if data is a 3d object, make it as a N^3*D sequence    %
%                  the same for the higher dim data                       %
%                  the input data S has been processed before input to fn %
%   SF: Scaled Factor of all states: if [], than return max(dist(S))      %
%   d: Distance option: 0: Euclidian 1:Manhattan 2:Cosine: if [], then 0  %
% - Output:                                                               %
%   rm: distance matrix                                                   %
%   SF: Scaled Factor of all states SF                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get number of states (rows)
    [n, ~] = size(S);
    
    % 1. Handle Distance Option 'd' (Default: Euclidean)
    if nargin < 3 || isempty(d)
        d = 0;
    end
    
    % 2. Calculate the raw distance matrix
    switch d
        case 0 % Euclidean
            dist_mat = squareform(pdist(S, 'euclidean'));
        case 1 % Manhattan
            dist_mat = squareform(pdist(S, 'cityblock'));
        case 2 % Cosine
            dist_mat = squareform(pdist(S, 'cosine'));
        otherwise
            error('Invalid distance option. Use 0, 1, or 2.');
    end
    
    % 3. Handle Scaling Factor 'SF' (Default: max distance)
    if nargin < 2 || isempty(SF)
        SF = max(dist_mat(:));
    end
    
    % 4. Prevent Division by Zero
    if SF == 0
        SF = 1; 
    end
    
    % 5. Calculate Similarity Matrix rhoM
    % Formula: rm(i,j) = 1 - dist(i,j) / SF
    rm = 1 - (dist_mat ./ SF);

end