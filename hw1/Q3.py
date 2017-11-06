import numpy as np
from Q1 import one_center_grid_coreset
import Q1
import matplotlib.pyplot as plt
import math


def far_dist(point, points):
    far_point = max(points, key=lambda p: Q1.dist(p, point))
    return Q1.dist(far_point, point)


errors = []
points = np.random.rand(2000, 3) * 1000
for e in np.arange(0.1, 1.1, 0.1):
    # a = raw_input()
    coreset_i = one_center_grid_coreset(points, e)
    coreset = points[coreset_i]

    Q = np.random.rand(100, 3) * 10 * 1000
    error = -float('inf')
    for q in Q:
        far_p = far_dist(q, points)
        far_c = far_dist(q, coreset)
        error = max(error, math.fabs(far_p - far_c) / far_p)
    errors.append(error)


plt.plot(np.arange(0.1, 1.01, 0.1), errors)
plt.xlabel('epsilon')
plt.ylabel('error')
plt.show()

