%% main.m
clc;
clear;
close all;
warning('off','all');

%% Initialize the problem and solve with CVX
% fundamental parameters
s = 5;                      % sparsity level
n = 20;                     % length of signal
rou = 0.1;                  % flip probability in the noise model

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

epsilon = 0.01;             % desired error bound
c = 50.0;                   % constant in determine the lower bound of m
C = 0.01;                   % constant in determine the upper bound of m
m = ceil(C * epsilon^(-2) * (rou - 0.5)^(-2) * ...
    s * log(2*n/s));        % required observation number
%lambda = 2 * sqrt(2 / pi) * (rou - 0.5);    % lambda for the noise model
delta = 8 * exp(- c * epsilon^2 * (rou - 0.5)^2 * m);
                            % the epsilon rate can be achieved with prob.
                            % 1 - delta

% randomly generate A and the observation y
A = normrnd(0, 1, [m, n]);
y = A * x;                  % true observation
y_c = y;                    % corrupted observation
flip_cnt = 0;
for i=1:m
    if rand() < rou
        y_c(i) = y(i) * (-1); % flip the sign of y with probability rou
        flip_cnt = flip_cnt + 1;
    end
end

% call CVX
cvx_begin
    variable x_p(n)
    maximize( y_c' * A * x_p )
    subject to
        abs( x_p ) <= sqrt(s)
        norm( x_p ) <= 1
cvx_end
err_cvx = norm(x - x_p)^2;
fprintf('theoretical error bound: %f\nexperimental error: %f\n', ...
    epsilon, err_cvx);

%% call DC for active learning
h = x;              % the hyperplane to be learned
K = 4;              % linear constant to calculate query times
debug = false;      % debug flag
h_p = DC(h, K, epsilon, delta, rou, debug);
err_dc = norm(h - h_p)^2;
fprintf('theoretical error bound: %f\nexperimental error: %f\n', epsilon, ...
    err_dc);

% end of main routine

