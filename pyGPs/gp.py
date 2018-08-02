import numpy

def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = numpy.where(distance==distance.min())
    return idy[0],idx[0]