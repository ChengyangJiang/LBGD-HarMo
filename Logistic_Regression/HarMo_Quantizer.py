def HarMo_Quantizer(x, m1, m2):
    K = 2 ** (m1 - 1)           
    Delta = 2 ** (-m2)          
    half = 0.5 * Delta
    y = np.clip(x, -K + half, K - half)
    q = np.sign(y) * Delta * (np.floor(np.abs(y) / Delta) + 0.5)
    q = np.clip(q, -K + half, K - half)
    return q
