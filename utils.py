import numpy as np
import time
import scipy.io
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def load(file_path_E, file_path_G, file_path_c, file_path_GI):
    data = scipy.io.loadmat(file_path_E)
    E = data['E'].transpose()
    E *= 2

    data = scipy.io.loadmat(file_path_G)
    G = data['G'].transpose()

    data = scipy.io.loadmat(file_path_c)
    center = data['c']

    data = scipy.io.loadmat(file_path_GI)
    GI = data['GI'].transpose()
    return G, E, center, GI, E.shape[1]

def check_half_space_intersection(G, beta_min, beta_max, c, b, adjusted_value):
    cG = G @ c
    min_value = np.sum(np.where(cG >= 0, beta_min * cG, beta_max * cG))
    return min_value + adjusted_value <= b

def generate_pascal_triangle(n):
    """ Generate Pascal's triangle up to the n+1-th row. (fix) """ 
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

def demo_result(dataset, methods, methods_exps, plot=True, fn=None):
    # Loop through all the methods, then for each method, plot all the experiments in a single plot
    # Total of num_methods * num_experiments colors
    fontsize = 20
    assert len(methods) == len(methods_exps), "Numer of methods and methods experiments do not match"
    if plot:
        plt.figure(figsize=(10, 6))
        #colors = plt.cm.viridis(np.linspace(0, 1, len(methods_exps) * len(methods_exps[0])))
        colors = ['r', 'b']
    # Loop through all the methods
    line_style = ['-', '--']
    for m, method in enumerate(methods):
        results_fn_prefix = f"{dataset}_{method}"
        if fn is None:
            results_fn = "./Results/" + results_fn_prefix + ".npz"
        else:
            results_fn = fn
        results = np.load(results_fn)['results']
        exps = methods_exps[m]
        # Plot the sum of memory over all experiments
        if plot:
            plt_x = np.arange(results.shape[0])
            sum_mem_result = np.sum(results[:, :, 1], axis=1) 
            plt.semilogy(plt_x, sum_mem_result, label=f'{method}_mem_sum', 
                        color=colors[m], linestyle=line_style[m])
        # Loop through all the experiments
        for exp_idx, exp in enumerate(exps):
            max_depth = exp[2]
            total_time = np.sum(results[:, exp_idx, 0])
            num_intersect = np.sum(results[:, exp_idx, 2])
            num_no_intersect = np.sum(results[:, exp_idx, 3])
            num_undecidable = np.sum(results[:, exp_idx, 4])
            print(f"For problem {results_fn_prefix}, with {exp_idx}-th experiment has the following results:")
            print(f"Total time: {total_time}s, Num Intersect: {num_intersect}, \
                Num No Intersect: {num_no_intersect}, Num Undecidable: {num_undecidable}, max depth: {max_depth}")
    if plot:
        # Label the plot
        plt.xlabel('Reachable Sets', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.ylabel('Memory Result', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        #plt.title(f'Memory usage comparison for {dataset}', fontsize=fontsize)
        plt.title(f'Memory usage comparison for Laub Loomis System', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.show()
    
        
    
