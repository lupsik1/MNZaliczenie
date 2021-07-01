import cupy as cp


def p_norm(v, p):
    vp = [x ** p for x in v]
    return sum(vp) ** (1 / p)


def get_random_point3d(min, max):
    p = [cp.random.uniform() * (max - min) + min for x in range(3)]
    return p
