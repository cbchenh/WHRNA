function [S_seq, indx] = flatten_to_sequence(data, D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FLATTEN_TO_SEQUENCE  Generalized data flattening to (TotalPoints x D)
%
% Inputs:
%   data : numeric/logical array of any shape:
%          - purely spatial: N1 x ... x Nk
%          - spatial + feature: N1 x ... x Nk x D
%   D    : number of features per point (expected feature dimension)
%
% Outputs:
%   S_seq : (TotalPoints x D) flattened sequence (double)
%   indx  : (TotalPoints x K) location indices (subscripts) for spatial dims
%           - if K==1: indx is (TotalPoints x 1)
%
% Notes:
%   - MATLAB's size() drops trailing singleton dimensions (e.g., N×N×1
%     prints as N×N). Therefore, when D==1 we treat ALL dimensions as spatial.
%   - For D>1, if the last dimension equals D, it is treated as the feature
%     axis; otherwise data is treated as already "flat" (all dims spatial).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % -------------------------
    % 0) Basic checks
    % -------------------------
    if nargin < 2
        error('flatten_to_sequence requires two inputs: data and D.');
    end
    if ~isscalar(D) || ~isnumeric(D) || ~isfinite(D) || D <= 0 || D ~= round(D)
        error('D must be a positive integer scalar.');
    end

    total_elements = numel(data);
    if mod(total_elements, D) ~= 0
        error('numel(data) (%d) must be divisible by D (%d).', total_elements, D);
    end
    num_points = total_elements / D;

    % -------------------------
    % 1) Identify spatial dimensions
    % -------------------------
    full_sz = size(data);

    if D == 1
        % Trailing singleton dims are not shown by size(), so treat all dims as spatial
        spatial_sz = full_sz;

    else
        % For D>1, check whether last dimension is truly the feature axis
        last_dim = size(data, ndims(data)); % safe for D>1 cases

        if last_dim == D && ndims(data) > 1
            spatial_sz = full_sz(1:end-1);
        else
            % Data does not end with D; treat entire array as spatial (already flat wrt features)
            spatial_sz = full_sz;
        end
    end

    % If spatial_sz is empty (possible if data is 1xD and treated as features only),
    % fall back to a 1D spatial index of length num_points.
    if isempty(spatial_sz)
        spatial_sz = num_points;
    end

    % -------------------------
    % 2) Reshape to (TotalPoints x D)
    % -------------------------
    S_seq = reshape(data, [num_points, D]);
    S_seq = double(S_seq);

    % -------------------------
    % 3) Compute spatial subscripts for each point
    % -------------------------
    num_spatial_dims = numel(spatial_sz);
    linear_idx = (1:num_points)';

    if num_spatial_dims == 1
        indx = linear_idx;
    else
        subs = cell(1, num_spatial_dims);
        [subs{:}] = ind2sub(spatial_sz, linear_idx);
        indx = cell2mat(subs);
    end
end
