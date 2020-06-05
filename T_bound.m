% Calculate the upper bound of queries in active learning
% Args:
%   epsilon: the upper bound of error
%   delta: estimation probability of 1 - delta
%   rou: flip probability
%
% Return:
%   T: the required number of queries

function T = T_bound(epsilon, delta, rou)
M = ceil(2*log(2/delta)/(-log(4*rou*(1-rou))));
T0 = 8*log(2/delta)/log(2*(1-rou));
T1 = 8*log(1/(8*pi*epsilon))/log(2*(1-rou));
T2 = (8/(log(2*(1-rou))))*(log(2*M) + log(4/(log(2*(1-rou)))));
T3 = (24*rou*log((1-rou)/rou)^2/log(2*(1-rou))^2)*(log(M) + log(4/delta));

T = ceil(real(M + max([T0, T1, T2, T3])));
end

