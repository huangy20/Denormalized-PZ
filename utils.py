import numpy as np
import time
import scipy.io
import os
import signal
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

# Define a handler that raises an exception when the time is exceeded
def timeout_handler(signum, frame):
    raise TimeoutError("Program exceeded the time limit.")

def get_dataset_formal_name(dataset):
    d = {
        'VanDelPol': 'Van der Pol Oscillator',
        'laubLoomis': 'Laub Loomis System'
    }
    return d[dataset]

def get_method_legend_name(method):
    d = {
        'depz' : "Denormalized PZ",
        'nmpz_dfs': "Cyclic PZ",
        'cora': 'CORA'
    }
    return d[method]

def demo_result_mem(dataset, methods, methods_exps, plot=True, fn=None, showcora=True):
    # Loop through all the methods, then for each method, plot all the experiments in a single plot
    # Total of num_methods * num_experiments colors
    fontsize = 20
    assert len(methods) == len(methods_exps), "Numer of methods and methods experiments do not match"
    if plot:
        plt.figure(figsize=(10, 6))
        #colors = plt.cm.viridis(np.linspace(0, 1, len(methods_exps) * len(methods_exps[0])))
        colors = ['r', 'b', 'g']
    # Loop through all the methods
    line_style = ['-', '--', '-.']
    mem_compare_candiate =[]
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
            max_mem = int(np.max(sum_mem_result))
            print(f"Data {dataset}, method {get_method_legend_name(method)} has max memory usage with {max_mem}\n")
            plt.semilogy(plt_x, sum_mem_result, label=f'{get_method_legend_name(method)}', 
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
                Num No Intersect: {num_no_intersect}, Num Undecidable: {num_undecidable}, max depth: {max_depth}\n")
    # Plot the core results:
    if showcora:
        cora_results_fn = f"./Results/cora_{dataset}_mem_time.npz"
        cora_results = np.load(cora_results_fn)['cora_result']
        if plot:
            plt_x = np.arange(cora_results.shape[0])
            cora_sum_mem_result = np.sum(cora_results[:, :, 1], axis=1)
            max_mem = int(np.max(cora_sum_mem_result))
            print(f"Data {dataset}, method {get_method_legend_name('cora')} has max memory usage with {max_mem}\n")
            plt.semilogy(plt_x, cora_sum_mem_result, label=f'{get_method_legend_name('cora')}', 
                        color=colors[2], linestyle=line_style[2])
    
    if plot:
        # Label the plot
        plt.xlabel('Steps', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.ylabel('Memory Result', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        #plt.title(f'Memory usage comparison for {dataset}', fontsize=fontsize)
        plt.title(f'Memory Usage Comparison for {get_dataset_formal_name(dataset)}', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.show()


def demo_result_time(dataset, methods, methods_exps, plot=True, fn=None, showcora=True):
    # Loop through all the methods, then for each method, plot all the experiments in a single plot
    # Total of num_methods * num_experiments colors
    fontsize = 20
    assert len(methods) == len(methods_exps), "Numer of methods and methods experiments do not match"
    if plot:
        plt.figure(figsize=(10, 6))
        #colors = plt.cm.viridis(np.linspace(0, 1, len(methods_exps) * len(methods_exps[0])))
        colors = ['r', 'b', 'g']
    # Loop through all the methods
    line_style = ['-', '--', '-.']
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
            sum_time_result = np.sum(results[:, :, 0], axis=1)
            plt.semilogy(plt_x, sum_time_result, label=f'{get_method_legend_name(method)}', 
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
                Num No Intersect: {num_no_intersect}, Num Undecidable: {num_undecidable}, max depth: {max_depth}\n")
    # Plot the core results:
    if showcora:
        cora_results_fn = f"./Results/cora_{dataset}_mem_time.npz"
        cora_results = np.load(cora_results_fn)['cora_result']
        if plot:
            plt_x = np.arange(cora_results.shape[0])
            cora_sum_time_result = np.sum(cora_results[:, :, 0], axis=1)
            plt.semilogy(plt_x, cora_sum_time_result, label=f'{get_method_legend_name('cora')}', 
                        color=colors[2], linestyle=line_style[2])
    if plot:
        # Label the plot
        plt.xlabel('Steps', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.ylabel('Computation Time Result', fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        #plt.title(f'Memory usage comparison for {dataset}', fontsize=fontsize)
        plt.title(f'Computation Time Comparison for {get_dataset_formal_name(dataset)}', fontsize=fontsize)
        #plt.legend(fontsize=fontsize, loc='lower right')
        plt.show()
    

def process_mat(fn='cora_van_mem_time.mat', dataset='VanDelPol'):
    from scipy.io import loadmat, savemat
    save_nm = f"cora_{dataset}_mem_time.npz"
    # Load the .mat file
    matlab_data = loadmat(fn)

    # Extract the matrix from the loaded data
    matrix_np = matlab_data['result_mat']
    np.savez(save_nm, cora_result=matrix_np)
    
    
