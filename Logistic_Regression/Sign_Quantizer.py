def Sign_Quantizer(X):

    X = np.asarray(X)
    d, n = X.shape
    Y = np.zeros((d, n))
    
    for j in range(n):
        x = X[:, j]
        norm1 = np.sum(np.abs(x))
        Y[:, j] = (norm1 / d) * np.where(x >= 0, 1, -1)
    
    return Y
