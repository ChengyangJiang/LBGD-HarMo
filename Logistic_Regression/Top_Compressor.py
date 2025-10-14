def Top_Compressor(X, alpha):

    X = np.asarray(X)
    d, n = X.shape
    Y = np.zeros((d, n))

    k = max(1, int(np.ceil(alpha * d)))

    for j in range(n):
        x = X[:, j]
        indexes = np.argsort(np.abs(x))[::-1]
        Y[indexes[:k], j] = x[indexes[:k]]

    q = Y
    return q
