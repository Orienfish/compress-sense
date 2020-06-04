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
%    sig_cnt = size(comb, 1);   % length of signal
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
%C = 4.5;
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
% fundamental parameters
s = 5;                      % sparsity level
n = 20;                     % length of signal
delta = 0.1;                % approximation level of using CVX
rou = 0.4;                  % flip probability in the noise model
C = 4.5;                    % constant in determine the lower bound of m
m = ceil(C * delta^(-2) * (rou - 0.5)^(-2) * ...
    s * log(2*n/s));        % required observation number
%lambda = 2 * sqrt(2 / pi) * (rou - 0.5);    % lambda for the noise model

% ramdomly generate the s-sparse signal with length n
comb = combnk(1:n, s);
% randomly select one comb as x
x = zeros(n, 1);
comb = comb(round(rand()*size(comb, 1)), :);
for k=1:s
    x(comb(k)) = -1 + 2*rand();
end
% normalize if necessary
if norm(x) > 1
    x = x / norm(x);
end

% randomly generate A and the observation y
A = -1 + 2 * rand(m, n);
y = A * x;                  % true observation
y_c = y;                    % corrupted observation
flip_cnt = 0;
for i=1:m
    if rand() < rou
        y_c(i) = y(i) * (-1); % flip the sign of y with probability rou
        flip_cnt = flip_cnt + 1;
    end
end
cvx_begin
    variable x_p(n)
    maximize( y_c' * A * x_p )
    subject to
        abs( x_p ) <= sqrt(s);
        norm( x_p ) <= 1;
cvx_end
err_cvx = norm(x - x_p)^2;