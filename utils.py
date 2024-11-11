import numpy as np
import time
import scipy.io
import os
from collections import defaultdict
import matplotlib.pyplot as plt

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

def print_result(check_result, i, time_usage):
    time_str = f" Time usage is: {time_usage}s."
    if check_result is True:
        print(f"Case {i}: Intersection found with the hyperplane." + time_str)
    elif check_result is False:
        print(f"Case {i}: No intersection found." + time_str)
    elif check_result is None:
        print(f"Case {i}: Inconclusive result - reached maximum recursion depth." + time_str)

def demo_result(dataset, methods, max_depth, dirs_with_bvals, plot=True):
     # Each result need to store: 0) computational time, 1) memory usage, 
    #  2) # intersections, 3) # no intersections, 4) # undecidedable cases
    if plot:
        plt.figure(figsize=(10, 6))
        colors = ['r', 'b']

    for m, method in enumerate(methods):
        results_fn_prefix = f"{dataset}_{method}_maxdepth_{max_depth}"
        results_fn = "./Results/" + results_fn_prefix + ".npz"
        results = np.load(results_fn)['results']
        if plot:
            plt_x = np.arange(results.shape[0])
           
        dirs_len = len(dirs_with_bvals)
        for dir_idx in range(dirs_len):
            total_time = np.sum(results[:, dir_idx, 0])
            num_intersect = np.sum(results[:, dir_idx, 2])
            num_no_intersect = np.sum(results[:, dir_idx, 3])
            num_undecidable = np.sum(results[:, dir_idx, 4])
            print(f"For problem {results_fn_prefix}, with {dir_idx}-th direction has the following results:")
            print(f"Total time: {total_time}s, Num Intersect: {num_intersect}, \
                Num No Intersect: {num_no_intersect}, Num Undecidable: {num_undecidable}")
            if plot:
                plt.plot(plt_x, results[:, dir_idx, 0], label=f'{method}_time', 
                         color=colors[m], linestyle='-')
                plt.plot(plt_x, results[:, dir_idx, 1], label=f'{method}_mem', 
                         color=colors[m], linestyle='--')
    
    if plot:
        # Label the plot
        plt.xlabel('Over All Sets')
        plt.ylabel('Time/Memory')
        plt.title('Time and memory usage comparison')
        plt.legend()
        plt.show()
    
