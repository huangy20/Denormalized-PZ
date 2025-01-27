from denormalized_pz import *
import sys

dataset = 'laubLoomis'
method= 'depz'

depz_exps =  [([0, 0, 0, 0, 0, 0, 1], 0.137, 40), \
 ([0, 0, 0, 0, 1, 0, 0], 0.0685, 40), \
 ([1, 0, 0, 0, 0, 0, 0], 0.457, 40), \
 ([0, -1, 0, 0, 0, 0, 0], -1.354, 40), \
 ([0, 0, -1, 0, 0, 0, 0], -1.6505, 40), \
 ([0, 0, 0, 0, 0, 1, 0], 0.045, 40)]

start_idx = 1
end_idx = 2000

result = exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, exps=depz_exps, dataset=dataset, save=False)

with open(f"Results/depz_laublommis_test_all.txt", 'w') as f:
    sys.stdout = f
    if not isinstance(result, str):
        result_fn = f"Results/{dataset}_{method}_table_exp.npz"
        np.savez(result_fn, results=result)
        demo_result_mem(dataset=dataset, methods=[method], methods_exps=[depz_exps], plot=False, fn=result_fn)
    else:
        print(result)

sys.stdout = sys.__stdout__

