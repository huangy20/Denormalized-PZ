import signal, sys, argparse
from denormalized_pz import *
from normalized_pz_dfs import *

# Example running command:  python3 timing_vandelpol_exp.py --exp_idx 0
parser = argparse.ArgumentParser(description="Run experiment with specified index and timeout.")
parser.add_argument('--exp_idx', type=int, default=1, help='Index of the experiment to run')
parser.add_argument('--timeout', type=int, default=900, help='Timeout for the experiment in seconds')
args = parser.parse_args()

# These are the only changing parameters
exp_idx = args.exp_idx
timeout = args.timeout

# Define a handler that raises an exception when the time is exceeded
def handler(signum, frame):
    raise TimeoutError("Program exceeded the time limit.")

# Set the signal alarm
signal.signal(signal.SIGALRM, handler)
dataset = 'VanDelPol'
method = 'nmpz_dfs'

nmpz_exps =  [([1, 0], -2.0165, 40), ([-1, 0], -2.138, 40), ([0, 1], -2.73, 40), ([0, -1], -2.804, 40)]

exp = [nmpz_exps[exp_idx]]
start_idx = 1
end_idx = 1349
result = None

try:
    signal.alarm(timeout)
    start_time = time.time()
    result = exp_nmpz_intersect_dfs(start_idx=start_idx, end_idx=end_idx,
                                    exps=exp, dataset=dataset, print=True, save=False)

except TimeoutError as e:
    print(e)
    result = f"Experiemnt {exp_idx} times out"

finally:
    # Disable the alarm
    end_time = time.time()
    usage_time = (end_time - start_time)
    signal.alarm(0)

with open(f"Results/nmpz_vanderpol_test_exp_{exp_idx}.txt", 'w') as f:
    sys.stdout = f
    if not isinstance(result, str):
        result_fn = f"Results/{dataset}_{method}_exp_{exp_idx}.npz"
        np.savez(result_fn, results=result)
        demo_result_mem(dataset=dataset, methods=['nmpz'], methods_exps=[exp], plot=False, fn=result_fn)
    else:
        print(result)
    print(f"Usage time is: {usage_time} seconds, dataset info: {dataset}, experiment info: {exp}")

# After the with block, standard output will return to the console
sys.stdout = sys.__stdout__
