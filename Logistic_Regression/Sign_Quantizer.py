def Sign_Quantizer(X):
    """
    Sign quantizer.
    Each column vector is scaled by its L1 norm / d, then replaced by ±1 signs.
    """
    X = np.asarray(X)
    d, n = X.shape
    Y = np.zeros((d, n))

    for j in range(n):
        x = X[:, j]
        norm1 = np.sum(np.abs(x))
        Y[:, j] = (norm1 / d) * np.where(x >= 0, 1, -1)

    q = Y
    return q
