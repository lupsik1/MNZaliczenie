import numpy as np
import matplotlib.pyplot as plt
import cupy as cp


def p_norm(v, p):
    vp = [x ** p for x in v]
    return sum(vp) ** (1 / p)


# ZAD 1
random_points = np.random.uniform(-1, 1, (1000, 3))

valid_points = [x for x in random_points if p_norm(x, 4) <= 1]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for p in valid_points:
    ax.scatter(p[0], p[1], p[2], marker='o')

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
