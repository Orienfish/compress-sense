%% main.m
clc;
clear;
close all;
warning('off','all')

%% generate s_ns and s_ns 1, 2, 3
%s = 5;
%n_list = round(linspace(10, 24, 8));     % the length of signal
%mwid_s = zeros(1, length(n_list));      % the squared mean width of s_ns
%mwid_s1 = zeros(1, length(n_list));     % the squared mean width of s_ns 1
%mwid_s2 = zeros(1, length(n_list));     % the squared mean width of s_ns 2
%mwid_s3 = zeros(1, length(n_list));     % the squared mean width of s_ns 3
%for i=1:length(n_list)
%    n = n_list(i);
    
    % ramdomly generate the s-sparse signal with length n
%    comb = combnk(1:n, s);
%    sig_cnt = size(comb, 1);   % count of potential signals in K
%    sig = zeros(sig_cnt, n);
%    sig1 = zeros(sig_cnt, n);
%    sig2 = zeros(sig_cnt, n);
%    sig3 = zeros(sig_cnt, n);
%    for j=1:sig_cnt
%        for k=1:s
%            sig(j, comb(j, k)) = -1 + 2*rand();
%            sig1(j, comb(j, k)) = max(0, sig(j, comb(j, k)));
%            sig2(j, comb(j, k)) = 1/s;
%            sig3(j, comb(j, k)) = 1/s * sign(sig(j, comb(j, k)));
%        end
        % normalize if necessary
%        if norm(sig(j, :)) > 1
%            sig(j, :) = sig(j, :) / norm(sig(j, :));
%        end
%        if norm(sig1(j, :)) > 1
%            sig1(j, :) = sig1(j, :) / norm(sig1(j, :));
%        end
%    end

    % calculate mean width w(K) for all potential signals in K
%    N_g = 100;        % number of random Gaussian variables in simulation
%    mwid_s(i) = mean_width(n, sig, N_g);
%    mwid_s1(i) = mean_width(n, sig1, N_g);
%    mwid_s2(i) = mean_width(n, sig2, N_g);
%    mwid_s3(i) = mean_width(n, sig3, N_g);
%end

%% plot the mean width of the s-sparse signal set versus various n
% together with lower and upper bound
%figure(1);
%c = 3.0;
C = 4.5;
%lb = sqrt(c * s * log(2*n_list/s));
%ub = sqrt(C * s * log(2*n_list/s));
%plot(n_list, mwid_s, '-*', 'DisplayName', 'w(S)', 'LineWidth', 2);
%hold on;
%plot(n_list, lb, '-s', 'DisplayName', 'Lower Bound', 'LineWidth', 2); 
%hold on;
%plot(n_list, ub, '-^', 'DisplayName', 'Upper Bound', 'LineWidth', 2);
%legend('location', 'northwest', 'FontSize', 16);
%xlabel('Length of Signal');
%ax = gca(); ax.FontSize = 16;

% plot the mean width of s_ns, s_ns 1, 2, 3
%figure(2);
%plot(n_list, mwid_s, '-*', 'DisplayName', 'w(S)', 'LineWidth', 2);
%hold on;
%plot(n_list, mwid_s1, '-s', 'DisplayName', 'w(S1)', 'LineWidth', 2);
%hold on;
%plot(n_list, mwid_s2, '-^', 'DisplayName', 'w(S2)', 'LineWidth', 2);
%hold on;
%plot(n_list, mwid_s3, '-d', 'DisplayName', 'w(S3)', 'LineWidth', 2);
%hold on;
%legend('location', 'northwest', 'FontSize', 16);
%xlabel('Length of Signal');
%ax = gca(); ax.FontSize = 16;

%% Use the convex program to decode the sensing data
% determine m
s = 5;
n = 20;
delta = 0.1;
m = ceil(C * delta^(-2) * s * log(2*n/s));
% randomly generate A
A = -1 + 2 * rand(m, n);