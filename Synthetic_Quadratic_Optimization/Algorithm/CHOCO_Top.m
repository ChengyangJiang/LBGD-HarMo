clc; 
clear;

%% ================== Basic Settings ==================
T = 50000;         % total iterations
n = 25;            % number of clients
d = 8;             % model dimension
zeta = 1;          % heterogeneity parameter
sigma_noise = 0.2; % Gaussian noise level (0 = none)
alpha  = 0.125;    
K      = max(1, floor(alpha*d)); % number of coordinates for Top-K
gamma  = 0.08;     
eta    = 1;        % initial step size (will be updated each iteration)

%% ================== Synthetic Least Squares ==================
Q = cell(n,1);
c = cell(n,1);
b = cell(n,1);

for i = 1:n
    Ai_scale = i / sqrt(n);
    bi = (zeta / i) * randn(d,1);
    b{i} = bi;
    Q{i} = (Ai_scale^2) * eye(d);
    c{i} = -(Ai_scale) * bi;
end

theta_star_local  = cell(n,1);
f_star_local      = zeros(n,1);
Q_sum = zeros(d,d); 
c_sum = zeros(d,1);

for i = 1:n
    theta_star_local{i} = - Q{i} \ c{i};
    f_star_local(i) = 0.5*theta_star_local{i}.'*Q{i}*theta_star_local{i} + c{i}.'*theta_star_local{i};
    Q_sum = Q_sum + Q{i}; 
    c_sum = c_sum + c{i};
end

theta_star_global = - Q_sum \ c_sum;
fprintf('=== Global optimum θ* ===\n'); 
disp(theta_star_global.');

%% ================== Topology ==================
A = generate_ring_graph(n);  % adjacency matrix
W = metropolis_from_adj(A);   % mixing matrix (symmetric, doubly stochastic)

%% ================== Initialization ==================
x = randn(d,n);    % local parameters
q = zeros(d,n);    % compressed difference
x_hat = zeros(d,n);% EF buffer
grad  = zeros(d,n);

% records
x_list     = zeros(d,n,T);
q_list     = zeros(d,n,T);
x_hat_list = zeros(d,n,T);
grad_list  = zeros(d,n,T);
opt_err_CHOCO_top  = zeros(1,T);

%% ---------------- Main Loop ----------------
for t = 1:T
    eta(t) = 0.2 / (t+1);    % diminishing step size
    if mod(t,1000) == 0
        fprintf('Iter %d / %d\n', t, T);
    end

    % local gradient updates
    for i = 1:n
        grad(:,i) = Q{i}*x(:,i) + c{i};
        if sigma_noise > 0
            grad(:,i) = grad(:,i) + sigma_noise*randn(d,1);
        end
        x(:,i) = x(:,i) - eta(t) * grad(:,i);
        q(:,i) = Top_alpha(x(:,i) - x_hat(:,i), K);
    end

    % error feedback update
    for j = 1:n
        x_hat(:,j) = q(:,j) + x_hat(:,j);
    end

    % consensus step
    for i = 1:n
        con = zeros(d,1);
        for j = 1:n
            con = con + W(i,j) * (x_hat(:,j) - x_hat(:,i));
        end
        x(:,i) = x(:,i) + gamma * con;
    end

    % evaluation: mean error w.r.t. θ*
    tmp = 0;

    for i = 1:n
        tmp = tmp + norm(x(:,i) - theta_star_global, 2);
    end
    opt_err_CHOCO_top(t) = tmp / n;
end

%% ---------------- Visualization ----------------
idx = 201:1:T;
figure;
semilogy(idx, opt_err_CHOCO_top(idx),'LineWidth',2); grid on;
xlim([0 50000]);
ylim([1e-4 1e0]);
xlabel('Iteration');
ylabel('Mean ||x_i - \theta^*||_2');
