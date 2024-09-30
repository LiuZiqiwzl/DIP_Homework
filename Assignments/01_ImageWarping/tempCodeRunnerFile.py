        alpha=1.0, eps=1e-8
        weights = np.zeros(len(points))
        p=[3,4]
        points = {[10, 10], [100, 100], [200, 50]}
        for i, point in enumerate(points):
            dist = np.linalg.norm(p - point)
            weights[i] = 1 / (dist ** (2 * alpha) + eps)
        print(weights)