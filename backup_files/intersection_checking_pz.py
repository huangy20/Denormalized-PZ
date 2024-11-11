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
    
    # Calculate the left, middle, and right coefficients
    left_coefficients = beta_min
    middle_coefficients = (beta_min + beta_max) / 2
    right_coefficients = beta_max
    
    # Expand each coefficient to match the shape of E
    left_expanded = np.tile(left_coefficients, (E.shape[1], 1)).T
    middle_expanded = np.tile(middle_coefficients, (E.shape[1], 1)).T
    right_expanded = np.tile(right_coefficients, (E.shape[1], 1)).T
    
    # Transform coefficients
    left_transformed = np.prod(left_expanded**E, axis=1)
    middle_transformed = np.prod(middle_expanded**E, axis=1)
    right_transformed = np.prod(right_expanded**E, axis=1)
    
    # Compute points
    left_point = np.dot(left_transformed, G) + adjusted_vector
    middle_point = np.dot(middle_transformed, G) + adjusted_vector
    right_point = np.dot(right_transformed, G) + adjusted_vector
    
    # Compute inner products with c
    left_inner_product = np.dot(left_point, c)
    middle_inner_product = np.dot(middle_point, c)
    right_inner_product = np.dot(right_point, c)
    
    # Determine the point with the smallest inner product
    min_inner_product = min(left_inner_product, middle_inner_product, right_inner_product)
    if min_inner_product == left_inner_product:
        return left_point
    elif min_inner_product == middle_inner_product:
        return middle_point
    else:
        return right_point


def check_half_space_intersection(G, beta_min, beta_max, c, b):
    cG = G @ c
    min_value = np.sum(np.where(cG >= 0, beta_min * cG, beta_max * cG))
    return min_value + adjusted_value <= b

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

def polynomial_zonotope_intersection(G, E, c, b_val, polyOrd=10, depth=0, max_depth=16, last_index=-1):
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
#folder_path = 'laubLoomis'
folder_path = 'VanDelPol'
iter = 0
# Start total runtime measurement
total_start_time = time.time()
check = 0 
# Iterate over all files in the folder
for i in range(1,1348):
    file_path_E = os.path.join(folder_path, f'E_interval_{i}.mat')
    file_path_G = os.path.join(folder_path, f'G_interval_{i}.mat')
    
    # Load G and E matrices from files
    file_path_E = os.path.join(folder_path, f'E_interval_{i}.mat')
    file_path_G = os.path.join(folder_path, f'G_interval_{i}.mat')
    file_path_c = os.path.join(folder_path, f'c_interval_{i}.mat')
    file_path_GI = os.path.join(folder_path, f'GI_interval_{i}.mat')
    # Load G and E matrices from files
    G, E, center,GI,size = load(file_path_E, file_path_G,file_path_c,file_path_GI)
    # Define the interval bounds a and b, and the half-space parameters c and b_val
    a = np.zeros(size)
    b = np.ones(size)
    c = np.array([1.0,0.0])
    b_val = 1.205
    # Experiment logic: Try edge cases and increment the maximum depth until (1) verify the safety set (2) the depths that the running times are below 60 second
    #(1,0, b = -5.06) 21.76(nv) vs 4.1(v) d = 12
    #(1,0, b = -5.08) 38.6996(nv) vs 0.3570(v) d = 12
    #(1,0, b = -5.1) 38.6996(nv) vs 0.3570(nv) d = 12
    #(-1.0,b = -5.38) no diff 
    #(0,1,b = -6.43) 61.0566(nv) vs  10.7119 (v) d = 12 vs d = 20
    #laubLoomis: [0.0,0.0,0.0,0.0,0.0,0.,1.0] b = -7.2 29.2(nv) 0.002 (v) d=12  vs d = 12 
    #            [0.0,0.0,0.0,0.0,1.0,0.0,0.0] b_val = -6.65 (31.2422) vs 0.0011(v) d= 12 vs d = 12
    #            [1.0,0.0,0.0,0.0,1.0,0.0,0.0] b_val = -13.167 53.5271(nv) vs 3.7695(v) d= 17 vs d = 40
    alpha = -np.sign(GI@ c)
    alpha_sum_vector = np.sum(alpha[:, np.newaxis] * GI, axis=0)
    adjusted_vector = alpha_sum_vector + center.reshape(-1) 
    adjusted_value = np.dot(adjusted_vector, c)
    check = polynomial_zonotope_intersection(G, E, c, b_val)
    if check is True:
        print(f"Case {i}: Intersection found with the hyperplane.")
        print(adjusted_value)
        break
        #
    elif check is False:
        pass
        #print(f"Case {i}: No intersection found after recursive splitting.")
    else:
        print(adjusted_value)
        iter += 1
        print(f"Case {i}: Inconclusive result - reached maximum recursion depth.")

# End total runtime measurement
total_end_time = time.time()
total_runtime = total_end_time - total_start_time

# Display total runtime
print(f"Total program runtime for all cases: {total_runtime:.4f} seconds")
print(iter)