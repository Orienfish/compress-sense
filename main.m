%% main.m
clc;
clear;
close all;
warning('off','all');

%% Initialize the problem and solve with CVX
% fundamental parameters
%s = 5;                      % sparsity level
s_list = [1, 5, 10, 15, 20];
times = 50;                % test times
err_mat_cvx = Inf(length(s_list), times);
err_mat_dc = Inf(length(s_list), times);

epsilon = 0.01;             % desired error bound
c = 50.0;                   % constant in determine the lower bound of m
C = 0.01;                   % constant in determine the upper bound of m
%lambda = 2 * sqrt(2 / pi) * (rho - 0.5);    % lambda for the noise model

for i = 1:length(s_list)
    s = s_list(i);              % sparsity level
    n = 20;                     % length of signal
    rho = 0.1;                  % flip probability in the noise model

    % ramdomly generate the s-sparse signal with length n
    comb = combnk(1:n, s);
    % randomly select one comb as x
    x = zeros(n, 1);
    sig_idx = randi([1 size(comb, 1)], 1, times); % generate times signals
    comb = comb(sig_idx, :);
        
    for j = 1:times
        % fill in the sparse signal
        for k=1:s
            x(comb(j, k)) = -1 + 2*rand();
        end
        % normalize if necessary
        if norm(x) > 1
            x = x / norm(x);
        end

        
        m = ceil(C * epsilon^(-2) * (rho - 0.5)^(-2) * ...
            s * log(2*n/s));        % required observation number
        delta = 8 * exp(- c * epsilon^2 * (rho - 0.5)^2 * m);
                                    % the epsilon rate can be achieved with prob.
                                    % 1 - delta

        % randomly generate A and the observation y
        A = normrnd(0, 1, [m, n]);
        y = A * x;                  % true observation
        y_c = y;                    % corrupted observation
        flip_cnt = 0;
        for k=1:m
            if rand() < rho
                y_c(k) = y(k) * (-1); % flip the sign of y with probability rho
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
        h_p = DC(h, K, epsilon, delta, rho, debug);
        err_dc = norm(h - h_p)^2;
        fprintf('theoretical error bound: %f\nexperimental error: %f\n', epsilon, ...
            err_dc);
        
        % update experiment records
        err_mat_cvx(i, j) = err_cvx;
        err_mat_dc(i, j) = err_dc;
    end
end

% plot
figure('Position', [0 0 500 375]);
plot(s_list, mean(err_mat_cvx, 2), '-*', 'DisplayName', 'CVX', 'LineWidth', 2);
hold on;
plot(s_list, mean(err_mat_dc, 2), '-s', 'DisplayName', 'DC', 'LineWidth', 2);
legend('FontSize', 16);
xlabel('Sparsity Level'); ylabel('Square Error');
ax = gca; ax.FontSize = 16;
grid on;

% end of main routine

