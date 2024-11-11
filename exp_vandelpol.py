from denormalized_pz import *
from normalized_pz_dfs import *

dataset = 'VanDelPol'
dirs_with_bvals = [([1.0, 0.0], -0.23)]
start_idx = 1
end_idx = 1349
# Note user is taking charge of setting end_idx to the real_end + 1

exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, dirs_with_bvals=dirs_with_bvals, 
                   dataset=dataset, max_depth=15)
exp_nmpz_intersect_dfs(start_idx=start_idx, end_idx=end_idx, dirs_with_bvals=dirs_with_bvals,
                       dataset=dataset, max_depth=15)