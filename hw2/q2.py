from q1 import *


def plot_points(points, color=None):
	plt.scatter(points[:, 0], points[:, 1], c=color)


def dist_to(points, p):
	return min([np.linalg.norm(x - p, ord=np.inf) for x in points])


def ab_approx(points, k, eps):
	centers = None
	while len(points) > 10:
		eps_coreset_idx = eps_sample(points, eps / k)
		eps_coreset = points[eps_coreset_idx]
		centers = np.concatenate((centers, eps_coreset)) if centers is not None else eps_coreset
		
		points = np.delete(points, eps_coreset_idx, axis=0)
		n = len(points)
		points_dist = [dist_to(eps_coreset, point) for point in points]
		points = points[np.argpartition(points_dist, n / 2)[:n / 2]]
	
	centers = np.concatenate((centers, points)) if centers is not None else points
	
	return centers


if __name__ == '__main__':
	dim = 2
	k = 2
	eps = 0.2
	points = np.random.rand(3000, dim) * 1000
	
	k_centers = ab_approx(points, k, eps)
	
	#
	#  testing
	#
	Q = [(get_square(dim) for i in range(k)) for i in range(100)]
	
	errl = []
	for q in Q:
		cs = 0.0
		for c in k_centers:
			cs += any(is_in_square(c, square) for square in q)
		cs /= k_centers.shape[0]
		ps = 0.0
		for p in points:
			ps += any(is_in_square(p, square) for square in q)
		ps /= points.shape[0]
		err = math.fabs(ps - cs)
		# print(err)
		errl.append(err)
	
	# print(errl)
	print('avg error: ' + str(np.average(errl)))
	print('max error: %f' % max(errl))
	print('min error: %f' % min(errl))
	print('coreset shape: ' + str(k_centers.shape))

	plt.scatter(points[:, 0], points[:, 1], label='original')
	plt.scatter(k_centers[:, 0], k_centers[:, 1], c='r', label='coreset')
	plt.legend()
	plt.show()

