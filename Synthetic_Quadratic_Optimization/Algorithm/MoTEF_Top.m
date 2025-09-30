clc; 
clear;

%% ---------------- Hyperparameters ----------------
T = 50000;        
n = 25;            % number of clients
d = 8;             % variable dimension
zeta = 1; 
sigma_noise = 0.2; % Gaussian noise added to gradients (0 means none)
alpha  = 0.125; 
K      = max(1, floor(alpha*d)); % number of coordinates kept in Top-K
gamma  = 0.2;     
eta    = 0.0005;   % gradient step size
lambda = 0.005;    % momentum parameter

%% ================== Synthetic Least Squares ==================
Q = cell(n,1);
c = cell(n,1);
b = cell(n,1);

for i = 1:n
    Ai_scale = i / sqrt(n);
    bi = (zeta / i) * randn(d,1);  % b_i ~ N(0, (zeta^2/i^2) I)
    b{i} = bi;
    Q{i} = (Ai_scale^2) * eye(d);  % A_i' A_i = (i^2/n) I_d
    c{i} = -(Ai_scale) * bi;      % -A_i' b_i = -(i/sqrt(n)) * b_i
end

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
fprintf('=== Global optimum θ* (least squares normal equation) ===\n'); 
disp(theta_star_global.');

%% ================== Topology ==================
A = generate_ring_graph(n);    
W = metropolis_from_adj(A);        

%% ================== Initialization ==================
X = ones(d,n);     
H = zeros(d,n);     % EF buffer for parameters
G = zeros(d,n);     % EF buffer for gradients
M = zeros(d,n);     % momentum tracking
V = zeros(d,n);     % gradient accumulation

% records
opt_err  = zeros(1,T); % optimality error
cons_err = zeros(1,T); % consensus error

%% ---------------- Main Loop ----------------
for t = 1:T
    % ---- 1) primal update
    X_new = X + gamma * H * (W - eye(n)) - eta * V;

    % ---- 2) parameter compression (error feedback)
    Qh = Top_alpha(X_new - H, K);
    H  = H + Qh;

    % ---- 3) momentum update with gradients
    M_new = (1 - lambda) * M;
    for i = 1:n
        grad_i = Q{i} * X_new(:,i) + c{i};
        if sigma_noise > 0
            grad_i = grad_i + sigma_noise * randn(d,1);
        end
        M_new(:,i) = M_new(:,i) + lambda * grad_i;
    end

    % ---- 4) gradient tracking update
    V_new = V + gamma * G * (W - eye(n)) + (M_new - M);

    % ---- 5) gradient compression (error feedback)
    Qg = Top_alpha(V_new - G, K);
    G  = G + Qg;

    % ---- 6) commit states
    X = X_new; 
    M = M_new; 
    V = V_new;

    % ====== evaluation (print every 1000 iterations) ======
    if mod(t,1) == 0
        tmp = 0;
        theta_bar = mean(X,2);
        cs = 0;
        for i = 1:n
            tmp = tmp + norm(X(:,i) - theta_star_global, 2);
            cs  = cs  + norm(X(:,i) - theta_bar, 2)^2;
        end
        opt_err(t)  = tmp / n;
        cons_err(t) = sqrt(cs);
        if mod(t,1000) == 0
        fprintf('Iter %6d | mean ||x_i-θ*|| = %.3e | cons = %.3e\n', ...
                t, opt_err(t), cons_err(t));
        end
    end
end
opt_err_MOTEF_top = opt_err;

idx = 201:1:T;
figure('Color','w'); 
semilogy(idx, opt_err(idx),'LineWidth',2); grid on;
xlabel('Iteration'); 
ylabel('Mean ||x_i - \theta^*||_2');
xlim([0 50000]);
ylim([1e-4 1e0]);


