function [A, L, G] = generate_er_graph(n, p, max_degree)
    % Generate a connected Erdős–Rényi (ER) graph with degree constraint

    isConnected = false;   % flag for connectivity
    while ~isConnected
        % initialize adjacency matrix
        A = zeros(n);

        % randomly add edges with degree constraint
        for i = 1:n
            for j = i+1:n
                if rand() < p && sum(A(i,:)) < max_degree && sum(A(j,:)) < max_degree
                    A(i,j) = 1;
                    A(j,i) = 1;  % undirected graph
                end
            end
        end

        % build graph object
        G = graph(A);

        % check connectivity
        if max(conncomp(G)) == 1
            isConnected = true;
        end
    end

    % compute Laplacian matrix L = D - A
    D = diag(sum(A, 2));  
    L = D - A;
end