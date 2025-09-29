function A = generate_ring_graph(n)
    % Generate adjacency matrix of a ring graph with n clients
    
    A = zeros(n);                 % initialize adjacency matrix
    for i = 1:n
        A(i, mod(i, n) + 1) = 1;  % connect client i to i+1
        A(mod(i, n) + 1, i) = 1;  % ensure symmetry (undirected graph)
    end
end