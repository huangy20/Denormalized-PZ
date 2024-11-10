import numpy as np
import time
import scipy.io
import os
from collections import defaultdict

def load(file_path_E,file_path_G):
    data = scipy.io.loadmat(file_path_E)
    E = data['E'].transpose()
    E *= 2

    data = scipy.io.loadmat(file_path_G)
    G = data['G'].transpose()
    return G,E,E.shape[1]

def check_half_space_intersection(G, beta_min, beta_max, c, b):
    '''
    '''
    cG = G @ c
    min_value = np.sum(np.where(cG >= 0, beta_min * cG, beta_max * cG))
    return min_value <= b

def generate_pascal_triangle(n):
    """ Generate Pascal's triangle up to the n-th row. """
    triangle = [[1]]
    for i in range(1, n+1):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

def print_result(check_result, i):
    if check_result is True:
        print(f"Case {i}: Intersection found with the hyperplane.")
    elif check_result is False:
        print(f"Case {i}: No intersection found.")
    elif check_result is None:
        print(f"Case {i}: Inconclusive result - reached maximum recursion depth.")        