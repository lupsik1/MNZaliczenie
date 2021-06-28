# ZADANIE 2
import numpy as np

a1 = np.array([1.0, 4.0, 3.0, 4.0])
a2 = np.array([4.0, 16.0, 12.0, 16.0])
a3 = np.array([3.0, 1.0, 3.0, 2.0])

M = np.vstack((a1, a2, a3)).T
print(np.linalg.svd(M))
