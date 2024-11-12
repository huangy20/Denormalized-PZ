from denormalized_pz import *
from normalized_pz_dfs import *

dataset = 'VanDelPol'
depz_exps = [([1.0, 0.0], -0.20165, 40)]
nmpz_exps = [([1.0, 0.0], -0.20165, 10)]
start_idx = 1
end_idx = 1349

# Note user is taking charge of setting end_idx to the real_end + 1
#'''
exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, exps=depz_exps, dataset=dataset)
exp_nmpz_intersect_dfs(start_idx=start_idx, end_idx=end_idx, exps=nmpz_exps, dataset=dataset)
#'''
# Present the results
demo_result(dataset=dataset, methods=['depz', 'nmpz_dfs'], methods_exps=[depz_exps, nmpz_exps], plot=True)
#'''

#demo_result(dataset=dataset, methods=['depz'], max_depths=max_depths, 
#            dirs_with_bvals=dirs_with_bvals, plot=True)