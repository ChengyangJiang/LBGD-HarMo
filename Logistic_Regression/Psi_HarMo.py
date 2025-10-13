def Psi_HarMo(t, d):
    i = np.arange(1, d + 1) 
    return np.sin(i * np.pi * t / (d+1))
