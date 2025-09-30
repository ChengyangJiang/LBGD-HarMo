clc; 
clear;

%% ================== Basic Settings ==================
T = 50000;               % total iterations
n = 25;                  % number of clients
d = 8;                   % model dimension
zeta = 1;                % heterogeneity parameter
sigma_noise = 0.2;       % Gaussian noise level (0 = none)
gamma  = 0.01;           
eta    = 1;              % initial step size (updated later)

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

% local/global analytical optima
theta_star_local  = cell(n,1);
f_star_local      = zeros(n,1);
Q_sum = zeros(d,d); 
c_sum = zeros(d,1);

for i = 1:n
    theta_star_local{i} = - Q{i} \ c{i};
    f_star_local(i) = 0.5 * theta_star_local{i}.' * Q{i} * theta_star_local{i} + ...
                      c{i}.' * theta_star_local{i};
    Q_sum = Q_sum + Q{i}; 
    c_sum = c_sum + c{i};
end

theta_star_global = - Q_sum \ c_sum;
fprintf('=== Global optimum θ* ===\n'); 
disp(theta_star_global.');

%% ================== Topology ==================
A = generate_ring_graph(n);  
W = metropolis_from_adj(A);   
L = laplacian(graph(A));       

%% ---------------- Initialization ----------------
x = randn(d,n);     % local parameters
grad  = zeros(d,n); % gradient buffer
opt_err_DSGD  = zeros(1,T); % optimality error record

%% ---------------- Main Loop ----------------
for t = 1:T
    eta(t) = 0.0036 / sqrt(t);  % diminishing step size
    if mod(t,1000) == 0
        fprintf('Iter %d / %d\n', t, T);
    end

    % consensus step: mix local states
    x_1 = x; % snapshot of previous state
    for i = 1:n
        con = zeros(d,1);
        for j = 1:n
            con = con + W(i,j) * x_1(:,j);
        end
        x(:,i) = con;
    end

    % local gradient update
    for i = 1:n
        grad(:,i) = Q{i} * x_1(:,i) + c{i};
        if sigma_noise > 0
            grad(:,i) = grad(:,i) + sigma_noise * randn(d,1);
        end
        x(:,i) = x(:,i) - eta(t) * grad(:,i);
    end

    % evaluate mean error w.r.t. θ*
    tmp = 0;
    for i = 1:n
        tmp = tmp + norm(x(:,i) - theta_star_global, 2);
    end
    opt_err_DSGD(t) = tmp / n;
end

%% ---------------- Visualization ----------------
figure;
semilogy(1:T, opt_err_DSGD(1:T), 'LineWidth', 2); grid on;
xlim([0 50000]);
ylim([1e-4 1e0]);
xlabel('Iteration');
ylabel('Mean ||x_i - \theta^*||_2');
