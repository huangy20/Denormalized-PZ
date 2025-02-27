from denormalized_pz import *
from normalized_pz_dfs import *

dataset = 'VanDelPol'
depz_exps =  [([1, 0], -2.0165, 10), ([-1, 0], -2.138, 10), ([0, 1], -2.73, 10), ([0, -1], -2.804, 10)]
nmpz_exps = [([1, 0], -2.0165, 10), ([-1, 0], -2.138, 10), ([0, 1], -2.73, 10), ([0, -1], -2.804, 10)]

start_idx = 1
end_idx = 1348 + 1

# Note user is taking charge of setting end_idx to the real_end + 1
exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, exps=depz_exps, dataset=dataset)
exp_nmpz_intersect_dfs(start_idx=start_idx, end_idx=end_idx, exps=nmpz_exps, dataset=dataset)

# Present the results
demo_result_mem(dataset=dataset, methods=['depz', 'nmpz_dfs'], methods_exps=[depz_exps, nmpz_exps], plot=True)
demo_result_time(dataset=dataset, methods=['depz', 'nmpz_dfs'], methods_exps=[depz_exps, nmpz_exps], plot=True)

