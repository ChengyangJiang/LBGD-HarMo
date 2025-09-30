clc; 
clear;

%% ---------------- Hyperparameters ----------------
T = 50000;                 % total iterations
n = 25;                    % number of clients
d = 8;                     % variable dimension

g0 = 10; gamma = 0.9999;   % scaling and decay for g_t
a = 1; kappa = 0.05;       % gradient step size
kappa0 = 0.005;            % consensus step size
eta = 5;                   % gradient scaling

% synthetic least-squares controls
zeta = 1;                  % heterogeneity: b_i ~ N(0, (zeta^2/i^2) I)
sigma_noise = 0.2;         % Gaussian noise level on gradients (0 = none)

%% ---------------- Synthetic Least Squares ----------------
Q = cell(n,1);
c = cell(n,1);
A_diag = zeros(n,1);       
b = cell(n,1);             

for i = 1:n
    Ai_scale = i / sqrt(n);        
    A_diag(i) = Ai_scale;
    bi = (zeta / i) * randn(d,1);   
    b{i} = bi;

    Q{i} = (Ai_scale^2) * eye(d);   % Q_i = A_i' A_i
    c{i} = -(Ai_scale) * bi;        % c_i = -A_i' b_i
end

theta_star_local = cell(n,1);
f_star_local     = zeros(n,1);

for i = 1:n
    theta_star_local{i} = - Q{i} \ c{i};
    f_star_local(i) = 0.5*theta_star_local{i}.'*Q{i}*theta_star_local{i} + c{i}.'*theta_star_local{i};
end

Q_sum = zeros(d,d); 
c_sum = zeros(d,1);

for i = 1:n
    Q_sum = Q_sum + Q{i}; 
    c_sum = c_sum + c{i};
end

theta_star_global = - Q_sum \ c_sum; 
fprintf('=== Global optimum θ* (least squares normal equation) ===\n');
disp(theta_star_global.');

%% ---------------- Topology ----------------
A = generate_cycle_graph(n);  
G_ring = graph(A);
L = laplacian(G_ring);         

figure; 
plot(G_ring,'Layout','circle'); 
title(sprintf('Cycle Graph (n=%d)',n));

for i = 1:n
    Nlist{i} = neighbors(G_ring,i); 
end

%% ---------------- Initialization ----------------
theta = ones(d,n);   
sigma = zeros(d,n);    
z     = zeros(d,n);    
grad  = zeros(d,n);    

%% ---------------- Main Loop ----------------
phi = zeros(d, T);                 % reserved for unified bound printing
x_hist     = zeros(d,n,T);         % history of x
theta_hist = zeros(d,n,T);         % history of θ

for t = 1:T
    if mod(t,1000)==0
        fprintf('Iter %d / %d\n', t, T);
    end

    g_t = g0 * (gamma^t);          
    x   = (theta - sigma) / g_t;   
    xhat = Sign_q(x);                

    for i = 1:n
        grad(:,i) = Q{i}*theta(:,i) + c{i};
        if sigma_noise > 0
            grad(:,i) = grad(:,i) + sigma_noise*randn(d,1);
        end
    end

    % record history
    x_hist(:,:,t)     = x;
    theta_hist(:,:,t) = theta;

    % sigma update
    sigma = sigma + kappa0 * g_t * xhat;

    % theta update
    theta = theta - kappa * ( sigma * L + (eta/((t+1)^a)) * grad );
end

%% ---------------- Evaluation & Visualization ----------------
% objective trajectory (constant terms omitted)
F_sum = zeros(1,T);
for t = 1:T
    val = 0;
    for i = 1:n
        th = theta_hist(:,i,t);
        val = val + 0.5*th.'*Q{i}*th + c{i}.'*th;
    end
    F_sum(t) = val;
end

% error metrics
opt_err_DHIC_sign = zeros(1,T);  % mean optimality error
cons_err          = zeros(1,T);  % consensus error
for t = 1:T
    tmp = 0;
    for i = 1:n
        tmp = tmp + norm(theta_hist(:,i,t) - theta_star_global, 2);
    end
    opt_err_DHIC_sign(t) = tmp / n;

    theta_bar = mean(theta_hist(:,:,t),2);
    cs = 0;
    for i = 1:n
        cs = cs + norm(theta_hist(:,i,t) - theta_bar,2)^2;
    end
    cons_err(t) = sqrt(cs);
end

idx = 1:1:T;
figure;
semilogy(idx, opt_err_DHIC_sign(idx),'LineWidth',2); grid on; hold on;
% semilogy(idx, cons_err(idx),'LineWidth',2);
xlim([0 50000]); 
ylim([1e-4 1e-1]); 

theta_final = theta;
theta_bar_final = mean(theta_final,2);
fprintf('\n||θ̄_T - θ*||_2 = %.6e\n', norm(theta_bar_final - theta_star_global, 2));
