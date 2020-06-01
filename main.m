%% main.m
clc;
clear;
close all;
warning('off','all')

% generate s_ns
s = 5;
n = 10;
comb = combnk(1:n, s);
sig_cnt = size(comb, 1);   % count of potential signals in K
sig = zeros(sig_cnt, n);
for i=1:sig_cnt
    sig(i, comb(i, :)) = 1;
end

% calculate mean width w(K) for all potential signals in K
N = 100;          % number of random Gaussian variables in simulation
rng('default')      % set random seed
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