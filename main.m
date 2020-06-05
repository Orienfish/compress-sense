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

epsilon = 0.01;             % approximation level of using CVX
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
%cvx_begin
%    variable x_p(n)
%    maximize( y_c' * A * x_p )
%    subject to
%        abs( x_p ) <= sqrt(s)
%        norm( x_p ) <= 1
%cvx_end
%err_cvx = norm(x - x_p)^2;
%fprintf('theoretical error bound: %f\nexperimental error: %f\n', ...
%    epsilon, err_cvx);

%% call DC for active learning
h = x;
h_cnt = length(h);
K = 4;              % linear constant to calculate query tims
h_p = zeros(h_cnt, 1);
h_norm = 1.0;
% start DC
for j=1:h_cnt-1
    if h_norm < epsilon
        break;  % all remaining dimensions are zeros
    end
    if j == h_cnt - 1 % last two elements
        h1 = h(end - 1);
        h2 = h(end);
    else
        e1 = zeros(h_cnt, 1);
        e1(j) = 1;
        h1 = e1' * h;
        e2 = zeros(h_cnt, 1);
        e2(j+1:end) = 1;
        h2 = norm(e2 .* h);
    end
    disp([h1, h2]);
    h_theta = DC2([h1; h2], K, epsilon, delta, rou);
    if h_theta == Inf
        fprintf('total probability in DC2 becomes invalid! reduce T!\n');
        break;
    end
    h_p(j) = h_norm * cos(h_theta);
    h_norm = h_norm * sin(h_theta);
    disp([h_p(j), h_norm]);
end
h_p(end) = h_norm;
err_dc = norm(h - h_p)^2;
fprintf('experimental error: %f\n', err_dc);

% end of main routine

% DC2 to estimate h: 1*2
function h_theta = DC2(h, K, epsilon, delta, rou)
    % create the node list for the ring
    prob_list = dlnode([0.0, 2*pi, 1/(2*pi)]);
    len_list = 1;
    h_theta = Inf;
    % calculate the query times
    T = ceil(K * (log(1/epsilon) + log(1/delta)));
    fprintf('T: %d\n', T);
    % start querying
    for m=1:T        
        fprintf('query #%d\n', m);
        display(prob_list, len_list)
        theta = eq_divide(prob_list, len_list);
        %fprintf('theta original: %f degrees\n', theta*180/pi);
        if theta == Inf
            return;             % return false, the total probability becomes invalid
        elseif theta > pi
            theta = theta - pi; % normalize to (0, pi]
        end
        %fprintf('theta: %f degrees\n', theta*180/pi);
        % find the orthogonal direction
        % add randomness to equally distribute the queries
        % theta_m locates in (-pi/2, 3pi/2]
        if rand() < 0.5
            theta_m = theta + pi/2;
        else
            theta_m = theta - pi/2;
        end
        %fprintf('theta_m: %f degrees\n', theta_m*180/pi);

        % find the point on the unit sphere and query its corrupted sign
        x_m = [cos(theta_m), sin(theta_m)];
        query = sign(x_m * h);
        if rand() < rou
            query = query * (-1);
        end
        if query >= 0
            w1 = 2 * (1 - rou); % update weight for R_plus
            w2 = 2 * rou;       % update weight for R_minus
        else
            w1 = 2 * rou;
            w2 = 2 * (1 - rou);
        end
        %fprintf('query: %d\n', query);
        %fprintf('R_plus weight: %f R_minus weight:%f\n', w1, w2);

        % update the probability dictionary
        R_plus_lb = theta_m - pi/2; % (-pi, pi]
        R_plus_ub = theta_m + pi/2; % (0, 2pi]
        %fprintf('R_plus_lb: %f degrees R_plus_ub: %f degrees\n', ...
        %    R_plus_lb*180/pi, R_plus_ub*180/pi);
        % add two new segments
        [prob_list, len_list] = add_node(prob_list, len_list, R_plus_lb);
        [prob_list, len_list] = add_node(prob_list, len_list, R_plus_ub);
        %fprintf('add two segments:\n');
        %display(prob_list, len_list);
        % update the probability
        prob_list = update_prob(prob_list, len_list, R_plus_lb, R_plus_ub, w1, w2);
        %fprintf('update the prob:\n');
        %display(prob_list, len_list);
    end
    % determine the estimation of h
    h_theta = find_h(prob_list, len_list);
end

function theta = eq_divide(prob_list, len_list)
    % accumulate the current probability until surpasses 0.5
    node = prob_list;
    cur_prob = 0.0;
    cur_idx = 1;
    while cur_idx <= len_list
        cur_prob = cur_prob + node.Data(3) * (node.Data(2) - node.Data(1));
        %fprintf('cur_prob: %f cur_idx: %d\n', cur_prob, cur_idx);
        if cur_prob >= 0.5
            break;
        end
        node = node.Next;
        cur_idx = cur_idx + 1;
    end
    fprintf('cur_prob: %f cur_idx: %d\n', cur_prob, cur_idx);
    if cur_idx > len_list
        theta = Inf;
    else
        % now we know the division takes place on segment cur_idx
        % we substract the surpassing probability and get the desired theta
        theta = node.Data(2) - (cur_prob - 0.5) / node.Data(3);
    end
end

% add another angle into the list
function [new_list, new_len] = add_node(prob_list, len_list, angle)
    % deal with the out-of-bound cases
    if angle < 0
        angle = angle + 2*pi;
    end
    node = prob_list;
    cur_idx = 1;
    while cur_idx <= len_list
        if node.Data(1) == angle
            break; % no new node to add
        end
        if node.Data(1) < angle && node.Data(2) > angle
            newnode = dlnode([angle, node.Data(2), node.Data(3)]);
            node.Data(2) = angle;
            newnode.insertAfter(node);
            len_list = len_list + 1;
            break;
        end
        node = node.Next;
        cur_idx = cur_idx + 1;
    end
    new_list = prob_list;
    new_len = len_list;
end

function new_list = update_prob(prob_list, len_list, lb, ub, w1, w2)
    node = prob_list;
    cur_idx = 1;
    while cur_idx <= len_list
        if lb >= 0
            if node.Data(1) >= lb && node.Data(2) <= ub
                % update R_plus
                node.Data(3) = node.Data(3) * w1;
            else
                % update R_minus
                node.Data(3) = node.Data(3) * w2;
            end
        else % R_plus of [lb, ub] overlaps the zero
            if node.Data(1) >= (lb+2*pi) || node.Data(2) <= ub
                % update R_plus
                node.Data(3) = node.Data(3) * w1;
            else
                % update R_minus
                node.Data(3) = node.Data(3) * w2;
            end
        end
        node = node.Next;
        cur_idx = cur_idx + 1;
    end
    new_list = prob_list;
end

function h_theta = find_h(prob_list, len_list)
    node = prob_list;
    cur_idx = 1;
    max_h.val = -Inf;
    max_h.lb = 0;
    max_h.ub = 0;
    while cur_idx <= len_list
        if node.Data(3) > max_h.val
            max_h.val = node.Data(3);
            max_h.lb = node.Data(1);
            max_h.ub = node.Data(2);
        end
        node = node.Next;
        cur_idx = cur_idx + 1;
    end
    h_theta = 0.5 * (max_h.lb + max_h.ub);
end

function display(prob_list, len_list)
    node = prob_list;
    cur_idx = 1;
    cur_prob = 0.0;
    while cur_idx <= len_list
        %fprintf("%d %f %f %f\n", cur_idx, node.Data(1)*180/pi, ...
        %    node.Data(2)*180/pi, node.Data(3));
        cur_prob = cur_prob + node.Data(3) * (node.Data(2) - node.Data(1));
        node = node.Next;
        cur_idx = cur_idx + 1;
    end
    fprintf('total probability is %f\n', cur_prob);
end
