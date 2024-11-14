from utils import *

def overapprox_nmpz(G, E):
    num_generators = G.shape[0]
    beta_min = np.full(num_generators, -1)
    beta_max = np.ones(num_generators)
    
    all_zeros = np.all(E == 0, axis=1)
    all_even = np.all(E % 2 == 0, axis=1) & ~all_zeros
    beta_min[all_zeros] = 1
    beta_min[all_even] = 0
    return beta_min, beta_max

def middle_point_polynomial_zonotope_normalized(G, E, adjusted_vector, dir):
    beta_min, beta_max = overapprox_nmpz(G, E)
    
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
    
    # Compute the three candidate points
    left_point = np.dot(left_transformed, G) + adjusted_vector
    middle_point = np.dot(middle_transformed, G) + adjusted_vector
    right_point = np.dot(right_transformed, G) + adjusted_vector
    
    # Compute inner products with c
    left_inner_product = np.dot(left_point, dir)
    middle_inner_product = np.dot(middle_point, dir)
    right_inner_product = np.dot(right_point, dir)
    
    # Determine the point with the smallest inner product
    min_inner_product = min(left_inner_product, middle_inner_product, right_inner_product)
    if min_inner_product == left_inner_product:
        return left_point
    elif min_inner_product == middle_inner_product:
        return middle_point
    else:
        return right_point

def removeRedundantExponents(E, G):
    if G.shape[0] == 0:
        return E, G
    # unique_exponents will be a stack of vectors corresponding to the same exponents (We are using them)
    # len(indices) == original number of generators. indices[i] == the first index of E which match this vector
    unique_exponents, indices = np.unique(E, axis=0, return_inverse=True)
    Gnew = np.zeros((unique_exponents.shape[0], G.shape[1]))
    for i, idx in enumerate(indices):
        Gnew[idx, :] += G[i, :]
    return unique_exponents, Gnew

def split_dependent_factor(G, E, ind, pascal_triangle):
    pascal = pascal_triangle
    G1, G2 = G.copy(), G.copy()
    E1, E2 = E.copy(), E.copy()
    
    for i in range(G.shape[0]):
        if E[i, ind] > 0:
            degree = E[i, ind]
            assert degree <= len(pascal), "Error, exist degree > len(pascal)"
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

def nmpz_intersection(G, E, c, b_val, pascal_triangle, adjusted_vector, adjusted_value, 
                      mem_track, depth=0, max_depth=15, last_index=-1):
    if depth > max_depth:  
        return None  
    # The overapproximation is correct with the following example
    # eg: g1*b1 = g1* 0.5(b2 + 1) where b1 \in [0, 1], b2 \in [-1, 1]
    # g1*b1 = 0.5*g1*b2 + 0.5*g1, when choose b2 == -1 RHS == 0 equivalent to choose b1 == 0 for LHS. 
    beta_min, beta_max = overapprox_nmpz(G, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val, adjusted_value=adjusted_value):
        return False

    middle_point = middle_point_polynomial_zonotope_normalized(G, E, adjusted_vector=adjusted_vector, dir=c)
    if np.dot(c, middle_point) <= b_val:
        return True

    split_index = (last_index + 1) % E.shape[1]
    G1, G2, E1, E2 = split_dependent_factor(G, E, split_index, pascal_triangle=pascal_triangle)
    
    # Add the memory usage information
    mem_track[0] += ((G1.shape[0] * G1.shape[1]) + (E1.shape[0] * E1.shape[1]))
    mem_track[0] += ((G2.shape[0] * G2.shape[1]) + (E2.shape[0] * E2.shape[1]))

    result_1 = nmpz_intersection(G1, E1, c, b_val, pascal_triangle, adjusted_vector, adjusted_value, 
                                 mem_track, depth + 1, max_depth, split_index)
    if result_1 is True:
        return True
    result_2 = nmpz_intersection(G2, E2, c, b_val, pascal_triangle, adjusted_vector, adjusted_value, 
                                 mem_track, depth + 1, max_depth, split_index)
    if result_2 is True:
        return True
    return None if result_1 is None or result_2 is None else False


def exp_timeout_nmpx_intersect_dfs(start_idx, end_idx, exps, dataset='VanDelPol', 
                                   do_print=True, save=True, filename=None, timeout=900):
    '''
    This is the experiment wrapper for doing intersection checking using normalized pz with dfs approach
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
    
    # Set the signal to trigger the handler if time is exceeded
    signal.signal(signal.SIGALRM, timeout_handler)

    # First loop through all the sets
    for i in range(start_idx, end_idx):
        set_idx = i - 1
        file_path_E = os.path.join(dataset, f'E_interval_{i}.mat')
        file_path_G = os.path.join(dataset, f'G_interval_{i}.mat')
        file_path_c = os.path.join(dataset, f'c_interval_{i}.mat')
        file_path_GI = os.path.join(dataset, f'GI_interval_{i}.mat')

        # Load G and E matrices from files and initialize the memory usage
        G, E, center, GI, size = load(file_path_E, file_path_G,file_path_c,file_path_GI)
        init_mem = int((G.shape[0] * G.shape[1]) + (E.shape[0] * E.shape[1]) + (GI.shape[0] * GI.shape[1]))
        
        # Since splitting would not change the max order, precompute the pascal triangle
        max_order_E = np.max(E)
        pascal_triangle = generate_pascal_triangle(max_order_E)

        # Loop through all the directions and corresponding extreme values
        for exp_idx, exp in enumerate(exps):
            mem_track = np.array([init_mem])

            # Extact the corresponding direction, b_val and max_depth
            c = np.array(exp[0])
            b_val = exp[1]
            max_depth = exp[2]
            
            # Prepare the adjusted vector and value due to GI and c
            alpha = -np.sign(GI@ c)
            alpha_sum_vector = np.sum(alpha[:, np.newaxis] * GI, axis=0)
            adjusted_vector = alpha_sum_vector + center.reshape(-1) 
            adjusted_value = np.dot(adjusted_vector, c)
            
            # Main starting part of each experiment of each set
            try:
                signal.alarm(timeout)
                start_time = time.time()
                check_result = nmpz_intersection(G, E, c, b_val, pascal_triangle=pascal_triangle,
                                                adjusted_vector=adjusted_vector, adjusted_value=adjusted_value, 
                                                max_depth=max_depth, mem_track=mem_track)
                # Here is the part where it successfully pass all the cases with a limited time FOR EACH CASE!
                end_time = time.time()
                time_usage = end_time - start_time
                if do_print:
                    print_result(check_result=check_result, i=i, time_usage=time_usage)
                results[set_idx, exp_idx, 0] = time_usage # Store the time
                results[set_idx, exp_idx, 1] = int(mem_track[0]) # Store the memory 
                # Store the corresponding number of results. 
                if check_result is True:
                    results[set_idx, exp_idx, 2] += 1
                elif check_result is False:
                    results[set_idx, exp_idx, 3] += 1
                elif check_result is None:
                    results[set_idx, exp_idx, 4] += 1
            except TimeoutError as e:
                # Case of the timeout
                if do_print:
                    print(f"Reachable set {i} times out with time limited for each set {timeout}")
                results[set_idx, exp_idx, 0] = time_usage # Store the time
                results[set_idx, exp_idx, 1] = int(mem_track[0]) # Store the memory
                results[set_idx, exp_idx, 4] += 1 # The amount of uncertain case increase 1
            finally:
                signal.alarm(0)
    if save:
        if filename is None:
            np.savez(f'Results/{dataset}_nmpz_dfs.npz', results=results)
        else:
            np.savez(filename, results=results)
    return results


def exp_nmpz_intersect_dfs(start_idx, end_idx, exps, dataset='VanDelPol', print=True, save=True, filename=None):
    '''
    This is the experiment wrapper for doing intersection checking using normalized pz with dfs approach
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
        file_path_E = os.path.join(dataset, f'E_interval_{i}.mat')
        file_path_G = os.path.join(dataset, f'G_interval_{i}.mat')
        file_path_c = os.path.join(dataset, f'c_interval_{i}.mat')
        file_path_GI = os.path.join(dataset, f'GI_interval_{i}.mat')

        # Load G and E matrices from files and initialize the memory usage
        G, E, center, GI, size = load(file_path_E, file_path_G,file_path_c,file_path_GI)
        init_mem = int((G.shape[0] * G.shape[1]) + (E.shape[0] * E.shape[1]) + (GI.shape[0] * GI.shape[1]))
        
        # Since splitting would not change the max order, precompute the pascal triangle
        max_order_E = np.max(E)
        pascal_triangle = generate_pascal_triangle(max_order_E)

        # Loop through all the directions and corresponding extreme values
        for exp_idx, exp in enumerate(exps):
            mem_track = np.array([init_mem])

            # Extact the corresponding direction, b_val and max_depth
            c = np.array(exp[0])
            b_val = exp[1]
            max_depth = exp[2]
            
            # Prepare the adjusted vector and value due to GI and c
            alpha = -np.sign(GI@ c)
            alpha_sum_vector = np.sum(alpha[:, np.newaxis] * GI, axis=0)
            adjusted_vector = alpha_sum_vector + center.reshape(-1) 
            adjusted_value = np.dot(adjusted_vector, c)
            
            # Main starting part of each experiment of each set
            start_time = time.time()
            check_result = nmpz_intersection(G, E, c, b_val, pascal_triangle=pascal_triangle,
                                             adjusted_vector=adjusted_vector, adjusted_value=adjusted_value, 
                                             max_depth=max_depth, mem_track=mem_track)
            end_time = time.time()
            time_usage = end_time - start_time
            if print:
                print_result(check_result=check_result, i=i, time_usage=time_usage)
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
        if filename is None:
            np.savez(f'Results/{dataset}_nmpz_dfs.npz', results=results)
        else:
            np.savez(filename, results=results)
    return results
