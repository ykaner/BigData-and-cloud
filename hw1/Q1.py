import numpy as np
import math
from scipy import misc
import cv2


def dist(x, y):
    return np.linalg.norm(x - y)


def get_block(point):
    point -= grid_start
    block = point // block_size
    res = []
    for i, b in enumerate(block):
        res.append(np.array([int(b)]))
    return res


def one_center_grid_coreset(points, epsilon):
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    d = points.shape[1]
    u = points[np.random.randint(len(points))]
    z = max(points, key=lambda p: dist(p, u))

    r = dist(u, z)
    global grid_start
    grid_start = u - r
    global block_size
    block_size = epsilon * r
    blocks = np.zeros([int(math.ceil(2/epsilon))]*d)
    coreset = []
    for i, p in enumerate(points):
        b_i = get_block(p)
        if blocks[b_i] == 0:
            blocks[b_i] = 1
            coreset.append(i)
    return coreset

