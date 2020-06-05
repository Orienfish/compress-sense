% DC in the active learning paper
% Args:
%   h: the hyperplane to be learned
%   K: linear constant to calculate query times
%   epsilon: desired error bound
%   delta: desired probability of 1-delta
%   rou: flip probability
%
% Return:
%   h_p: the estimated hyperplane

function h_p = DC(h, K, epsilon, delta, rou)
h_cnt = length(h);
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
end

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