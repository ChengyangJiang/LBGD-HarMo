function A = generate_torus_graph(n)
    % Generate adjacency matrix of an n×n torus graph

    N = n * n;              % total number of clients
    A = zeros(N);           % initialize adjacency matrix

    % Map 2D coordinate (row, col) to 1-based index
    idx = @(row, col) mod(row-1, n)*n + mod(col-1, n) + 1;

    for row = 1:n
        for col = 1:n
            u = idx(row, col);

            % Connect to four neighbors with wrap-around
            v1 = idx(row+1, col);   % down
            v2 = idx(row-1, col);   % up
            v3 = idx(row, col+1);   % right
            v4 = idx(row, col-1);   % left

            A(u, v1) = 1; A(v1, u) = 1;
            A(u, v2) = 1; A(v2, u) = 1;
            A(u, v3) = 1; A(v3, u) = 1;
            A(u, v4) = 1; A(v4, u) = 1;
        end
    end
end