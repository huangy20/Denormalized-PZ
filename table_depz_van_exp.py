from denormalized_pz import *
import sys

dataset = 'VanDelPol'
method= 'depz'

depz_exps =  [([1, 0], -2.0165, 40), ([-1, 0], -2.138, 40), ([0, 1], -2.73, 40), ([0, -1], -2.804, 40)]

start_idx = 1
end_idx = 1349

result = exp_depz_intersect(start_idx=start_idx, end_idx=end_idx, exps=depz_exps, dataset=dataset, save=False)

# Present the results to a desired file
with open(f"Results/depz_vanderpol_test_all.txt", 'w') as f:
    sys.stdout = f
    if not isinstance(result, str):
        result_fn = f"Results/{dataset}_{method}_table_exp.npz"
        np.savez(result_fn, results=result)
        demo_result_mem(dataset=dataset, methods=[method], methods_exps=[depz_exps], plot=False, fn=result_fn)
    else:
        print(result)


# After the with block, standard output will return to the console
sys.stdout = sys.__stdout__