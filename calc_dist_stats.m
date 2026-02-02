function [stats, names] = calc_dist_stats(data, prefix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CALC_DIST_STATS
% Computes 8 statistical descriptors for a given data distribution.
%
% Inputs:
%   data   : Vector of numeric data (e.g., degrees, centralities)
%   prefix : String prefix for header names (e.g., 'Global_Degree')
%
% Outputs:
%   stats  : 1x8 vector [Min, Q1, Median, Q3, Max, IQR, Mean, Std]
%   names  : 1x8 cell array of feature names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Define the 8 statistical features we want
    suffixes = {'Min', 'Q1', 'Median', 'Q3', 'Max', 'IQR', 'Mean', 'Std'};
    % Create full header names (e.g., 'Global_Degree_Min')
    names = cellfun(@(s) [prefix, '_', s], suffixes, 'UniformOutput', false);

    if isempty(data)
        stats = zeros(1, 8); 
        return;
    end

    % Remove NaNs to ensure robust statistics
    data = data(~isnan(data));
    
    if isempty(data)
        % If data was all NaNs, return zeros
        stats = zeros(1, 8);
    else
        % Calculate Quantiles
        q = quantile(data, [0.25, 0.5, 0.75]);
        
        % [Min, Q1, Median, Q3, Max, IQR, Mean, Std]
        stats = [min(data), q(1), q(2), q(3), max(data), iqr(data), mean(data), std(data)];
    end
end