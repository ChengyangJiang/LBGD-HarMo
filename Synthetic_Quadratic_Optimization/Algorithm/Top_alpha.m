function Y = Top_alpha(X, K)
    [d, n] = size(X);
    Y = zeros(d,n);
    if K >= d, Y = X; return; end
    for j = 1:n
        x = X(:,j);
        [~, idx] = maxk(abs(x), K);
        y = zeros(d,1); 
        y(idx) = x(idx);
        Y(:,j) = y;
    end
end
