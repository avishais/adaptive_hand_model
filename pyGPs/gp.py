import numpy as np
from sklearn.neighbors import KDTree #pip install -U scikit-learn

# def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
#     distance = (y_array-y_point)**2 + (x_array-x_point)**2
#     idy,idx = numpy.where(distance==distance.min())
#     return idy[0],idx[0]

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [-2, 1], [3, 2]])
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print kdt.query(np.array([[1, 0.9]]), k=3, return_distance=True)  # returns the indices