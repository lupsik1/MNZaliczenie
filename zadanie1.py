import numpy as np
from toolkit import *
import matplotlib.pyplot as plt

# ZAD 1
random_points = [get_random_point3d(-1, 1) for x in range(1000)]

valid_points = [x for x in random_points if p_norm(x, 4) <= 1]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for p in valid_points:
    ax.scatter(p[0], p[1], p[2], marker='o')

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


