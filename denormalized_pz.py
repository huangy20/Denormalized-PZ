from utils import *

def overapprox_depz(G, a, b, E):
    """
    Since a, b are row vectors, a_exp = np.power(a, E) == a**E
    a_exp[:, i] represent factor a_i with some power over all the generators
    So doing np.prod over axis=1 (column) will give the min / max value 
        for each term through interval arithmetic eg: evaluate a_1 * a_2^2
    Note: a, b be all positive is important to maintain the correctness of interval arithmetic
    So our work assume postive range which can be obtained through simple shift at the beginning
    of the splitting. Such shifting does not involved in the afterwards splitting process
    """
    a_exp = np.power(a, E)
    b_exp = np.power(b, E)
    beta_min = np.prod(a_exp, axis=1)
    beta_max = np.prod(b_exp, axis=1)
    return beta_min, beta_max

def middle_point_polynomial_zonotope_with_dom(G, a, b, E, adjusted_vector, dir):
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
    left_inner_product = np.dot(left_point, dir)
    middle_inner_product = np.dot(middle_point, dir)
    right_inner_product = np.dot(right_point, dir)
    
    # Find the point with the smallest inner product
    min_inner_product = min(left_inner_product, middle_inner_product, right_inner_product)
    if min_inner_product == left_inner_product:
        return left_point
    elif min_inner_product == middle_inner_product:
        return middle_point
    else:
        return right_point

def depz_intersection(G, a, b, E, c, b_val, adjusted_vector, adjusted_value, mem_track, 
                      depth=0, max_depth=15, last_index=-1, tolerance=1e-7):
    if depth > max_depth:
        return None
    beta_min, beta_max = overapprox_depz(G, a, b, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val, adjusted_value=adjusted_value):
        return False
    
    # Check whether there exist a point in the set already intersect the hyperplane
    middle_point = middle_point_polynomial_zonotope_with_dom(G, a, b, E, adjusted_vector=adjusted_vector)
    if np.dot(c, middle_point) <= b_val:
        return True
    
    # Do the cyclic splitting here
    split_index = (last_index + 1) % len(a)

    # Do the splitting here
    split_a_1, split_b_1 = a.copy(), b.copy()
    split_b_1[split_index] = (a[split_index] + b[split_index]) / 2
    split_a_2, split_b_2 = a.copy(), b.copy()
    split_a_2[split_index] = (a[split_index] + b[split_index]) / 2
    
    # Update the memory for having the new splitted range
    mem_track[0] += (split_a_1.shape[0] + split_b_1.shape[0])
    mem_track[0] += (split_a_2.shape[0] + split_b_2.shape[0])

    result_1 = depz_intersection(G, split_a_1, split_b_1, E, c, b_val, adjusted_vector, adjusted_value,
                                 mem_track, depth + 1, max_depth, split_index, tolerance)
    if result_1 is True:
        return True
    result_2 = depz_intersection(G, split_a_2, split_b_2, E, c, b_val, adjusted_vector, adjusted_value, 
                                 mem_track, depth + 1, max_depth, split_index, tolerance)
    return None if result_1 is None or result_2 is None else False



def exp_depz_intersect(start_idx, end_idx, exps, dataset='VanDelPol', print=True, save=True):
    '''
    This is the experiment wrapper for doing intersection checking using denormalized pz 
    over all the directions.
    We measure:
        1) number of intersection, no intersection, not decidable cases
        2) computational time
        3) memory and space usage 
    Args:
        dataset: the data set one would like to do the intersection checking experiment
        start_idx, end_idx: the start and end index (end_idx excluded) to loop over 
                            the user takes charge of matching this with the file names
        exps: a list of tuples of experiment informations (requires unique directions)
            (list representing the direction, corresponding b value, max_depth information)
    '''

    num_sets = end_idx - start_idx
    num_dirs = len(exps)
    
    # Each result need to store: 0) computational time, 1) memory usage, 
    #  2) # intersections, 3) # no intersections, 4) # undecidedable cases
    results = np.zeros((num_sets, num_dirs, 5))
    # First loop through all the sets
    for i in range(start_idx, end_idx):
        set_idx = i - 1
        # Load the reachable sets
        file_path_E = os.path.join(dataset, f'E_interval_{i}.mat')
        file_path_G = os.path.join(dataset, f'G_interval_{i}.mat')
        file_path_c = os.path.join(dataset, f'c_interval_{i}.mat')
        file_path_GI = os.path.join(dataset, f'GI_interval_{i}.mat')

        # Load G, GI and E matrices and center from files and initialize the memory usage
        G, E, center, GI, alpha_size = load(file_path_E, file_path_G, file_path_c, file_path_GI)
        init_mem = int((G.shape[0] * G.shape[1]) + (E.shape[0] * E.shape[1]) + (GI.shape[0] * GI.shape[1]))

        # Loop through all the directions and corresponding extreme values
        for exp_idx, exp in enumerate(exps):
            # Initialize the track variables
            mem_track = np.array([init_mem])

            # Extract the corresponding direction, b_val and max_depth
            c = np.array(exp[0])
            b_val = exp[1]
            max_depth = exp[2]

            # Define the interval bounds a and b, and the half-space parameters c and b_val
            a = np.zeros(alpha_size)
            b = np.ones(alpha_size)

            # Prepare the adjusted vector and value due to GI and c
            alpha = -np.sign(GI@ c) # For all positive generators choose -1, else choose 1
            # Note: the alpha_sum_vector might not be the minimum over all directions
            # But! It must be the min over the direction c
            alpha_sum_vector = np.sum(alpha[:, np.newaxis] * GI, axis=0)
            # Need to add the center together with the independent part
            adjusted_vector = alpha_sum_vector + center.reshape(-1) 
            # Get the projected value (should be min value) of the adjusted vector over direction c
            adjusted_value = np.dot(adjusted_vector, c) 
            
            # Main starting part of each experiment of each set
            start_time = time.time()
            check_result = depz_intersection(G, a, b, E, c, b_val, adjusted_vector=adjusted_vector,
                                             adjusted_value=adjusted_value, max_depth=max_depth, 
                                             mem_track=mem_track)
            end_time = time.time()
            time_usage = end_time - start_time
            if print:
                print_result(check_result=check_result, i=i, time_usage=time_usage)
            
            # Store the result information
            results[set_idx, exp_idx, 0] = time_usage # Store the time
            results[set_idx, exp_idx, 1] = int(mem_track[0]) # Store the memory 
            
            # Store the corresponding number of results. 
            if check_result is True:
                results[set_idx, exp_idx, 2] += 1
            elif check_result is False:
                results[set_idx, exp_idx, 3] += 1
            elif check_result is None:
                results[set_idx, exp_idx, 4] += 1
    if save:
        np.savez(f'Results/{dataset}_depz.npz', results=results)
    return results