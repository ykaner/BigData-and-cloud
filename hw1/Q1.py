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


if __name__ == '__main__':
    points = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [2.01, 4.01, 5.99],
        [5, 9, 34],
        [5, 9, 34],
        [35, 4, 43]
    ])
    # points = misc.face()
    points = cv2.imread("C:\\Users\\ykane\\Pictures\\IMG_0709.JPG")
    cv2.imshow('ps2', points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points = points.reshape(points.shape[0] * points.shape[1] / 2, points.shape[2] * 2)
    points = points.astype(float)
    print(points.shape)
    coreset = one_center_grid_coreset(points, 0.1)
    print('coreset: ' + str(coreset))
    print('len(coreset): %d' % len(coreset))
    # testing:

    q = np.random.rand(6) * 250
    print('q: ' + str(q))
    m_c = 0
    for c in coreset:
        m_c = max(dist(points[c], q), m_c)
    print('m_c: ' + str(m_c))
    m_p = 0
    for p in points:
        m_p = max(dist(p, q), m_p)
    print('m_p: ' + str(m_p))

    print(m_c / m_p)
    print(m_p / m_c)
