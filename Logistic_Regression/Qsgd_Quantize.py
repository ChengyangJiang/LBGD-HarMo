def Qsgd_Quantize(x, d, is_biased):
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    scale = 1
    if is_biased:
        n = len(x)
        scale = 1. / (np.minimum(n / d ** 2, np.sqrt(n) / d) + 1.)
    return scale * np.sign(x) * norm * new_level / d
