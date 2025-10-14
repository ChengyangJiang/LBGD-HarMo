def HarMo_Quantizer(X, m1, m2):

    X = np.asarray(X)
    d, n = X.shape
    Y = np.zeros((d, n))

    # Quantization step
    K = 2 ** (m1 - 1)           
    Delta = 2 ** (-m2)          
    half = 0.5 * Delta

    for j in range(n):
        x = X[:, j]
        y = np.clip(x, -K + half, K - half)
        q_col = np.sign(y) * Delta * (np.floor(np.abs(y) / Delta) + 0.5)
        q_col = np.clip(q_col, -K + half, K - half)
        Y[:, j] = q_col

    q = Y
    return q
