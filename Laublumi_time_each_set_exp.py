import signal, sys, argparse
from denormalized_pz import *
from normalized_pz_dfs import *

'''
This experiment is set up for timing over each single set, not all the sets. Not used in the paper 
'''

# Example running command:  python3 Laublumi_time_each_set_exp.py --exp_idx 0 --timeout 5
parser = argparse.ArgumentParser(description="Run experiment with specified index and timeout.")
parser.add_argument('--exp_idx', type=int, default=1, help='Index of the experiment to run')
parser.add_argument('--timeout', type=int, default=900, help='Timeout for the experiment in seconds')
args = parser.parse_args()

# These are the only changing parameters
exp_idx = args.exp_idx
timeout = args.timeout

# Set the signal alarm
dataset = 'laubLoomis'
method = 'nmpz_dfs'

nmpz_exps = [([0, 0, 0, 0, 0, 0, 1], 0.137, 40), \
([0, 0, 0, 0, 1, 0, 0], 0.0685, 40), \
([1, 0, 0, 0, 0, 0, 0], 0.457, 40), \
([0, -1, 0, 0, 0, 0, 0], -1.354, 40), \
([0, 0, -1, 0, 0, 0, 0], -1.6505, 40),\
([0, 0, 0, 0, 0, 1, 0], 0.045, 40)]

exp = [nmpz_exps[exp_idx]]
start_idx = 1
end_idx = 2000


result_fn = f"Results/{dataset}_{method}_exp_{exp_idx}.npz"
result = exp_timeout_nmpx_intersect_dfs(start_idx=start_idx, end_idx=end_idx, 
                                        exps=exp, dataset=dataset, do_print=True, 
                                        save=True, filename=result_fn, timeout=timeout)

with open(f"Results/nmpz_laublumi_test_exp_{exp_idx}.txt", 'w') as f:
    sys.stdout = f
    demo_result_mem(dataset=dataset, methods=[method], methods_exps=[exp], plot=False, fn=result_fn)

# After the with block, standard output will return to the console
sys.stdout = sys.__stdout__
demo_result_mem(dataset=dataset, methods=[method], methods_exps=[exp], plot=True, fn=result_fn)
