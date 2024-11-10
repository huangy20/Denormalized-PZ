from utils import *

def overapprox_depz(G, a, b, E):
    a_exp = np.power(a, E)
    b_exp = np.power(b, E)
    beta_min = np.prod(a_exp, axis=1)
    beta_max = np.prod(b_exp, axis=1)
    return beta_min, beta_max

def middle_point_polynomial_zonotope_with_dom(G, a, b, E):
    middle_coefficients = (a + b) / 2
    transformed_coefficients = np.prod(middle_coefficients**E, axis=1)
    return np.dot(transformed_coefficients, G)

def depz_intersection(G, a, b, E, c, b_val, mem_track, 
                      depth=0, max_depth=15, last_index=-1, tolerance=1e-12):
    if depth > max_depth:
        return None
    beta_min, beta_max = overapprox_depz(G, a, b, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val):
        return False
    middle_point = middle_point_polynomial_zonotope_with_dom(G, a, b, E)
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
    
    # Update the memory for having the new splitted range TODO: Think about if we should add 2 or 2n
    mem_track[0] += (split_a_1.shape[0] + split_b_1.shape[0])
    mem_track[0] += (split_a_2.shape[0] + split_b_2.shape[0])

    result_1 = depz_intersection(G, split_a_1, split_b_1, E, c, b_val, mem_track, 
                                 depth + 1, max_depth, split_index, tolerance)
    if result_1 is True:
        return True
    result_2 = depz_intersection(G, split_a_2, split_b_2, E, c, b_val, mem_track, 
                                 depth + 1, max_depth, split_index, tolerance)
    return None if result_1 is None or result_2 is None else False



def exp_depz_intersect(start_idx, end_idx, dirs_with_bvals, dataset='VanDelPol', max_depth=15, print=True):
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
        dirs_with_bvals: a list of tuples of (list representing the direction, corresponding b value)
    '''

    num_sets = end_idx - start_idx
    num_dirs = len(dirs_with_bvals)
    
    # Each result need to store: 0) computational time, 1) memory usage, 
    #  2) # intersections, 3) # no intersections, 4) # undecidedable cases
    results = np.zeros((num_sets, num_dirs, 5))
    # First loop through all the sets
    for i in range(start_idx, end_idx):
        set_idx = i - 1
        file_path_E = os.path.join(dataset, f'E_interval_{i}.mat')
        file_path_G = os.path.join(dataset, f'G_interval_{i}.mat')

        # Load G and E matrices from files and initialize the memory usage
        G, E, size = load(file_path_E, file_path_G)
        init_mem = int((G.shape[0] * G.shape[1]) + (E.shape[0] + E.shape[1]))

        # Loop through all the directions and corresponding extreme values
        for dir_w_b_idx, dir_w_b in enumerate(dirs_with_bvals):
            # Initialize the track variables
            mem_track = np.array([init_mem])

            # Extact the corresponding direction and b_val
            c = dir_w_b[0]
            b_val = dir_w_b[1]
            # Define the interval bounds a and b, and the half-space parameters c and b_val
            a = np.zeros(size)
            b = np.ones(size)
            start_time = time.time()
            check_result = depz_intersection(G, a, b, E, c, b_val, 
                                             max_depth=max_depth, mem_track=mem_track)
            end_time = time.time()
            if print:
                print_result(check_result=check_result, i=i)
            results[set_idx, dir_w_b_idx, 0] = end_time - start_time # Store the time
            results[set_idx, dir_w_b_idx, 1] = int(mem_track[0]) # Store the memory 
            # Store the corresponding number of results. 
            if check_result is True:
                results[set_idx, dir_w_b_idx, 2] += 1
            elif check_result is False:
                results[set_idx, dir_w_b_idx, 3] += 1
            elif check_result is None:
                results[set_idx, dir_w_b_idx, 4] += 1
    return results