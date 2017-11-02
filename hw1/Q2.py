import numpy as np
from Q1 import one_center_grid_coreset
import matplotlib.pyplot as plt


coreset_lens = []

for n in range(100, 2001, 100):
    points = np.random.rand(n, 3) * 1000
    coreset = one_center_grid_coreset(points, 0.5)
    coreset_lens.append(len(coreset))

plt.plot(range(100, 2001, 100), coreset_lens)
plt.xlabel('dataset size')
plt.ylabel('coreset size')
plt.show()
