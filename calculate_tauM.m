function taum = calculate_tauM(rm, gm, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tauM(i,j) = rm(i,j)^(lambda/(1-lambda) * gm(i,j)^((1-lambda)/lambda))   %
% - Input:                                                                %
%   rm: rm sequence N*N                                                   %
%   gm: gm sequence N*N                                                   %
%   lambda: 0<lambda<1, and by default lambda is 0.5                      %
% - Output:                                                               %
%   tauM: tau sequence                                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 1. Handle default lambda
    if nargin < 3 || isempty(lambda)
        lambda = 0.5;
    end

    % 2. Check if sizes of rm and gm are the same
    if ~isequal(size(rm), size(gm))
        error('Dimension mismatch: rm is %dx%d but gm is %dx%d. They must be the same size.', ...
            size(rm,1), size(rm,2), size(gm,1), size(gm,2));
    end

    % 3. Validate lambda range
    if lambda <= 0 || lambda >= 1
        error('Lambda must be strictly between 0 and 1.');
    end

    % 4. Calculate exponents
    exp1 = lambda / (1 - lambda);
    exp2 = (1 - lambda) / lambda;

    % 5. Compute tauM using element-wise operations
    % .^ and .* ensure the calculation is performed on each (i,j) pair
    taum = (rm.^exp1) .* (gm.^exp2);
end
