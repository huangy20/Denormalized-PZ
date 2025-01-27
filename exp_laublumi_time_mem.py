from denormalized_pz import *
from normalized_pz_dfs import *

dataset = 'laubLoomis'

depz_exps =  [([0, 0, 0, 0, 0, 0, 1], 0.137, 3), \
 ([0, 0, 0, 0, 1, 0, 0], 0.0685, 3), \
 ([1, 0, 0, 0, 0, 0, 0], 0.457, 3), \
 ([0, -1, 0, 0, 0, 0, 0], -1.354, 3), \
 ([0, 0, -1, 0, 0, 0, 0], -1.6505, 3), \
 ([0, 0, 0, 0, 0, 1, 0], 0.045, 3)]

nmpz_exps = [([0, 0, 0, 0, 0, 0, 1], 0.137, 3), \
 ([0, 0, 0, 0, 1, 0, 0], 0.0685, 3), \
 ([1, 0, 0, 0, 0, 0, 0], 0.457, 3), \
 ([0, -1, 0, 0, 0, 0, 0], -1.354, 3), \
 ([0, 0, -1, 0, 0, 0, 0], -1.6505, 3), \
 ([0, 0, 0, 0, 0, 1, 0], 0.045, 3)]


start_idx = 1
end_idx = 2000 + 1

# Note user is taking charge of setting end_idx to the real_end + 1
exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, exps=depz_exps, dataset=dataset)
exp_nmpz_intersect_dfs(start_idx=start_idx, end_idx=end_idx, exps=nmpz_exps, dataset=dataset)

# Present the results
demo_result_mem(dataset=dataset, methods=['depz', 'nmpz_dfs'], methods_exps=[depz_exps, nmpz_exps], plot=True)
demo_result_time(dataset=dataset, methods=['depz', 'nmpz_dfs'], methods_exps=[depz_exps, nmpz_exps], plot=True)

