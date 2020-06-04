% Calculate the mean width of signal by simulation
% Args:
%   n: the number of elements in the signal
%   sig: each row represents a potential sparse signal
%   N: the number of random Gaussian variables in the simulation
%
% Return:
%   mwid: the simulated mean width of the signal

function mwid = mean_width(n, sig, N)
sig_cnt = size(sig, 1);   % count of potential signals
rng('default')            % set random seed
r = normrnd(0, 1, [N, n]);
sum = 0.0;
for i=1:N
    cur_max = -Inf;
    cur_min = Inf;
    for j=1:sig_cnt
        prod = r(i, :) * sig(j, :)';
        cur_max = max(cur_max, prod);
        cur_min = min(cur_min, prod);
    end
    sum = sum + cur_max - cur_min;
end
mwid = sum / N;
end

