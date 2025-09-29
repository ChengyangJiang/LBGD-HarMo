function A = generate_fully_graph(n)
    % Generate adjacency matrix of a fully connected graph with n clients

    % initialize n×n matrix with ones
    A = ones(n);

    % remove self-connections (set diagonal to zero)
    A(1:n+1:end) = 0;
end