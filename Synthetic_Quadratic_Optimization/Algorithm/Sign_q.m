function Y = Sign_q(X)
    [d, n] = size(X);
    Y = zeros(d, n);
    for j = 1:n
        x = X(:, j);
        norm1 = sum(abs(x));
        Y(:, j) = (norm1 / d) * sign(x);
    end
end
