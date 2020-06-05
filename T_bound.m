% Calculate the upper bound of queries in active learning
% Args:
%   epsilon: the upper bound of error
%   delta: estimation probability of 1 - delta
%   K: constant parameter
%
% Return:
%   T: the required number of queries

function T = T_bound(epsilon, delta, K)
T = ceil(K * (log(1/epsilon) + log(1/delta)));
end

