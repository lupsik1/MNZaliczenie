import cupy as cp
import numba
# zadanie 3 wykonane za pomocą układu GPU za pomocą bibliotekii cupy

def t3():
    n = 100000
    x = cp.random.randn(n).astype(cp.float32)
    y = cp.random.randn(n).astype(cp.float32)

    print(x)