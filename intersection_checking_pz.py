import numpy as np
import time
import scipy.io
from load_data import load
import os 
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

def overapproximate_polynomial_zonotope(G, E):
    num_generators = G.shape[0]
    beta_min = np.full(num_generators, -1)
    beta_max = np.ones(num_generators)
    
    all_zeros = np.all(E == 0, axis=1)
    all_even = np.all(E % 2 == 0, axis=1) & ~all_zeros
    beta_min[all_zeros] = 1
    beta_min[all_even] = 0
    
    return beta_min, beta_max

def middle_point_polynomial_zonotope(G, E):
    beta_min, beta_max = overapproximate_polynomial_zonotope(G, E)
    middle_coefficients = (beta_min + beta_max) / 2
    middle_coefficients_expanded = np.tile(middle_coefficients, (E.shape[1], 1)).T
    transformed_coefficients = np.prod(middle_coefficients_expanded**E, axis=1)
    return np.dot(transformed_coefficients, G)

def check_half_space_intersection(G, beta_min, beta_max, c, b):
    cG = G @ c
    min_value = np.sum(np.where(cG >= 0, beta_min * cG, beta_max * cG))
    return min_value <= b

def removeRedundantExponents(E, G):
    if G.shape[0] == 0:
        return E, G
    unique_exponents, indices = np.unique(E, axis=0, return_inverse=True)
    Gnew = np.zeros((unique_exponents.shape[0], G.shape[1]))
    for i, idx in enumerate(indices):
        Gnew[idx, :] += G[i, :]
    return unique_exponents, Gnew

def split_dependent_factor(G, E, ind, polyOrd):
    pascal = generate_pascal_triangle(polyOrd)
    G1, G2 = G.copy(), G.copy()
    E1, E2 = E.copy(), E.copy()
    
    for i in range(G.shape[0]):
        if E[i, ind] > 0:
            degree = E[i, ind]
            if degree <= polyOrd:
                for k in range(degree + 1):
                    coeff1 = pascal[degree][k] * 0.5 ** degree
                    coeff2 = coeff1 * (-1 if (degree - k) % 2 != 0 else 1)
                    new_gen1 = G[i, :] * coeff1
                    new_gen2 = G[i, :] * coeff2
                    if k != degree:
                        G1 = np.vstack((G1, new_gen1))
                        G2 = np.vstack((G2, new_gen2))
                        E1 = np.vstack((E1, E[i, :]))
                        E2 = np.vstack((E2, E[i, :]))
                        E1[-1, ind] = k
                        E2[-1, ind] = k
                    else:
                        G1[i, :] = new_gen1
                        G2[i, :] = new_gen2
    E1, G1 = removeRedundantExponents(E1, G1)
    E2, G2 = removeRedundantExponents(E2, G2)
    return G1, G2, E1, E2

def polynomial_zonotope_intersection(G, E, c, b_val, polyOrd=12, depth=0, max_depth=10, last_index=-1):
    if depth > max_depth:  
        return None  

    beta_min, beta_max = overapproximate_polynomial_zonotope(G, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val):
        return False

    middle_point = middle_point_polynomial_zonotope(G, E)
    if np.dot(c, middle_point) <= b_val:
        return True

    split_index = (last_index + 1) % E.shape[1]
    G1, G2, E1, E2 = split_dependent_factor(G, E, split_index, polyOrd)

    result_1 = polynomial_zonotope_intersection(G1, E1, c, b_val, polyOrd, depth + 1, max_depth, split_index)
    if result_1 is True:
        return True
    result_2 = polynomial_zonotope_intersection(G2, E2, c, b_val, polyOrd, depth + 1, max_depth, split_index)
    if result_2 is True:
        return True

    return None if result_1 is None or result_2 is None else False

# Folder path containing the .mat files
folder_path = 'VanDelPol'
iter = 0
# Start total runtime measurement
total_start_time = time.time()
check = 0 
# Iterate over all files in the folder
for i in range(1, 1349):
    file_path_E = os.path.join(folder_path, f'E_interval_{i}.mat')
    file_path_G = os.path.join(folder_path, f'G_interval_{i}.mat')
    
    # Load G and E matrices from files
    G, E, size = load(file_path_E, file_path_G)
    
    # Define the interval bounds a and b, and the half-space parameters c and b_val
    c = np.array([1.0, 0.0])
    b_val = -0.23
    
    check = polynomial_zonotope_intersection(G, E, c, b_val)
    if check is True:
        print(f"Case {i}: Intersection found with the hyperplane.")
        break
        #
    elif check is False:
        pass
        #print(f"Case {i}: No intersection found after recursive splitting.")
    else:
        iter += 1
        print(f"Case {i}: Inconclusive result - reached maximum recursion depth.")

# End total runtime measurement
total_end_time = time.time()
total_runtime = total_end_time - total_start_time

# Display total runtime
print(f"Total program runtime for all cases: {total_runtime:.4f} seconds")
print(iter)