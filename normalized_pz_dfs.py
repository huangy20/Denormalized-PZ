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

def middle_point_polynomial_zonotope_normalized(G, E):
    beta_min, beta_max = overapprox_nmpz(G, E)
    middle_coefficients = (beta_min + beta_max) / 2
    middle_coefficients_expanded = np.tile(middle_coefficients, (E.shape[1], 1)).T
    transformed_coefficients = np.prod(middle_coefficients_expanded**E, axis=1)
    return np.dot(transformed_coefficients, G)

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

def nmpz_intersection(G, E, c, b_val, mem_track, polyOrd=12, depth=0, max_depth=15, last_index=-1):
    if depth > max_depth:  
        return None  

    beta_min, beta_max = overapprox_nmpz(G, E)
    if not check_half_space_intersection(G, beta_min, beta_max, c, b_val):
        return False

    middle_point = middle_point_polynomial_zonotope_normalized(G, E)
    if np.dot(c, middle_point) <= b_val:
        return True

    split_index = (last_index + 1) % E.shape[1]
    G1, G2, E1, E2 = split_dependent_factor(G, E, split_index, polyOrd)
    
    # Add the memory usage information
    mem_track[0] += ((G1.shape[0] * G1.shape[1]) + (E1.shape[0] * E1.shape[1]))
    mem_track[0] += ((G2.shape[0] * G2.shape[1]) + (E2.shape[0] * E2.shape[1]))

    result_1 = nmpz_intersection(G1, E1, c, b_val, mem_track, polyOrd, depth + 1, max_depth, split_index)
    if result_1 is True:
        return True
    result_2 = nmpz_intersection(G2, E2, c, b_val, mem_track, polyOrd, depth + 1, max_depth, split_index)
    if result_2 is True:
        return True
    return None if result_1 is None or result_2 is None else False




def exp_nmpz_intersect_dfs(start_idx, end_idx, dirs_with_bvals, 
                           dataset='VanDelPol', max_depth=15, print=True, save=True):
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
            c = np.array(dir_w_b[0])
            b_val = dir_w_b[1]
            # Define the interval bounds a and b, and the half-space parameters c and b_val
            a = np.zeros(size)
            b = np.ones(size)
            start_time = time.time()
            check_result = nmpz_intersection(G, E, c, b_val, max_depth=max_depth, mem_track=mem_track)
            end_time = time.time()
            time_usage = end_time - start_time
            if print:
                print_result(check_result=check_result, i=i, time_usage=time_usage)
            results[set_idx, dir_w_b_idx, 0] = time_usage # Store the time
            results[set_idx, dir_w_b_idx, 1] = int(mem_track[0]) # Store the memory 
            # Store the corresponding number of results. 
            if check_result is True:
                results[set_idx, dir_w_b_idx, 2] += 1
            elif check_result is False:
                results[set_idx, dir_w_b_idx, 3] += 1
            elif check_result is None:
                results[set_idx, dir_w_b_idx, 4] += 1
    if save:
        np.savez(f'Results/{dataset}_nmpz_dfs_maxdepth_{max_depth}.npz', results=results)
    return results
