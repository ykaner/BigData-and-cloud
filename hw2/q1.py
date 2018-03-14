import math
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_point(dim=3, low=0, high=255):
	return np.random.randint(low, high, dim)


def get_square(dim=3):
	p1 = get_point(dim)
	p2 = get_point(dim, low=max(p1))
	return np.array([p1, p2])


def is_in_square(point, square):
	square = square.tolist() if type(square) is np.ndarray else square
	point = point.tolist()
	return square[0] <= point < square[1]


points = None


def eps_sample(pts, epsilon):
	points = pts
	_eps_sample.orig_points = points
	return _eps_sample(range(len(pts)), epsilon, 0)


def _eps_sample(idx, epsilon, dim=0):
	if type(idx) is not np.ndarray:
		idx = np.array(idx)
	
	n = len(idx)
	k = int(math.ceil(epsilon * n))
	# p_sorted = sorted(orig_points, key=lambda x: x[dim])
	
	p_idx = [x[dim] for x in _eps_sample.orig_points[idx]]
	if k < n:
		p_idx = np.argpartition(p_idx, kth=range(0, n - 1, k))
	else:
		p_idx = np.argsort(p_idx)
	# p_sorted = np.array(p_sorted, object)
	# p_sorted = np.partition(p_sorted, kth=range(0, n-1, k), axis=0)
	p_sorted = idx[p_idx]
	lines = []
	for i in xrange(0, n, k):
		lines.append(np.array(p_sorted[i:i + k]))
	
	coreset = []
	if dim == _eps_sample.orig_points.shape[1] - 1:
		for line in lines:
			coreset.append(np.random.choice(line))
	
	else:
		for line in lines:
			coreset += _eps_sample(line, epsilon, dim + 1)
	
	return coreset


if __name__ == '__main__':
	dim = 2
	points = np.random.rand(1000, dim) * 1000
	points = points.astype(float)
	
	coreset = eps_sample(points, 0.1)
	coreset = points[coreset]
	coreset = np.array(coreset)
	print(coreset.shape)
	
	#
	# checking
	#
	Q = [get_square(dim) for i in range(100)]
	
	errl = []
	for q in Q:
		cs = 0.0
		for c in coreset:
			cs += is_in_square(c, q)
		cs /= coreset.shape[0]
		ps = 0.0
		for p in points:
			ps += is_in_square(p, q)
		ps /= points.shape[0]
		err = math.fabs(ps - cs)
		# print(err)
		errl.append(err)
	
	# print(errl)
	print('avg error: ' + str(np.average(errl)))
	print('max error: %f' % max(errl))
	print('min error: %f' % min(errl))
	
	plt.scatter(points[:, 0], points[:, 1], label='original')
	plt.scatter(coreset[:, 0], coreset[:, 1], c='r', label='coreset')
	plt.legend()
	plt.show()
