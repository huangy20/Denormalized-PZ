import numpy as np
import time
import scipy.io
import os 
from load_data import load
def overapproximate_polynomial_zonotope(G, a, b, E):
    a_exp = np.power(a, E)
    b_exp = np.power(b, E)
    beta_min = np.prod(a_exp, axis=1)
    beta_max = np.prod(b_exp, axis=1)
    return beta_min, beta_max

def middle_point_polynomial_zonotope(G, a, b, E):
    # Calculate the middle coefficients
    middle_coefficients = (a + b) / 2
    
    # Calculate transformed coefficients for left, middle, and right
    left_transformed = np.prod(a**E, axis=1)
    middle_transformed = np.prod(middle_coefficients**E, axis=1)
    right_transformed = np.prod(b**E, axis=1)
    
    # Compute the points
    left_point = np.dot(left_transformed, G) + adjusted_vector
    middle_point = np.dot(middle_transformed, G) + adjusted_vector
    right_point = np.dot(right_transformed, G) + adjusted_vector
    
    # Compute the inner products with vector c
    left_inner_product = np.dot(left_point, c)
    middle_inner_product = np.dot(middle_point, c)
    right_inner_product = np.dot(right_point, c)
    
    # Find the point with the smallest inner product
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

def polynomial_zonotope_intersection(G, a, b, E, c, b_val, depth=0, max_depth=20,last_index=-1, tolerance=1e-7):
    if depth > max_depth:
        return None
    beta_min, beta_max = overapproximate_polynomial_zonotope(G, a, b, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val):
        return False
    middle_point = middle_point_polynomial_zonotope(G, a, b, E)
    if np.dot(c, middle_point) <= b_val:
        return True
    split_index = (last_index + 1) % len(a)
    if abs(b[split_index] - a[split_index]) < tolerance:
        middle_point[split_index] = (a[split_index] + b[split_index]) / 2
        return np.dot(c, middle_point) <= b_val

    split_a_1, split_b_1 = a.copy(), b.copy()
    split_b_1[split_index] = (a[split_index] + b[split_index]) / 2
    split_a_2, split_b_2 = a.copy(), b.copy()
    split_a_2[split_index] = (a[split_index] + b[split_index]) / 2
    result_1 = polynomial_zonotope_intersection(G, split_a_1, split_b_1, E, c, b_val, depth + 1, max_depth, split_index, tolerance)
    if result_1 is True:
        return True
    result_2 = polynomial_zonotope_intersection(G, split_a_2, split_b_2, E, c, b_val, depth + 1, max_depth, split_index, tolerance)
    return None if result_1 is None or result_2 is None else False

# Folder path containing the .mat files
folder_path = 'laubLoomis'
folder_path = 'VanDelPol'
check = 0
# Start total runtime measurement
total_start_time = time.time()
iter = 0
# Iterate over all files in the folder
for i in range(1,1348):
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
    b_val = -2
    alpha = -np.sign(GI@ c) # For all positive generators choose -1, else choose 1
    alpha_sum_vector = np.sum(alpha[:, np.newaxis] * GI, axis=0)
    adjusted_vector = alpha_sum_vector + center.reshape(-1) # This vector guarantee to minize on dir of c 
    adjusted_value = np.dot(adjusted_vector, c)
    # Check intersection with the half-space
    check = polynomial_zonotope_intersection(G, a, b, E, c, b_val)
    if check is True:
        print(f"Case {i}: Intersection found with the hyperplane.")
        break
    elif check is False:
        pass
        #print(adjusted_value)
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
