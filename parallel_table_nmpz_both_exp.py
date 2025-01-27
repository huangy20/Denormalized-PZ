import subprocess

'''
Running single_table_nmpz_van_exp.py and single_table_nmpz_laub_exp.py in parallel
'''

programs = [
    ["python3", "single_table_nmpz_van_exp.py", "--exp_idx", "0", "--timeout", "900"],
    ["python3", "single_table_nmpz_van_exp.py", "--exp_idx", "1", "--timeout", "900"],
    ["python3", "single_table_nmpz_van_exp.py", "--exp_idx", "2", "--timeout", "900"],
    ["python3", "single_table_nmpz_van_exp.py", "--exp_idx", "3", "--timeout", "900"],

    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "0", "--timeout", "900"],
    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "1", "--timeout", "900"],
    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "2", "--timeout", "900"],
    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "3", "--timeout", "900"],
    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "4", "--timeout", "900"],
    ["python3", "single_table_nmpz_laub_exp.py", "--exp_idx", "5", "--timeout", "900"],
]

processes = []
for program in programs:
    try:
        print(f"Starting: {' '.join(program)}")
        process = subprocess.Popen(program)
        processes.append(process)
    except Exception as e:
        print(f"Failed to start: {' '.join(program)}")
        print(f"Error: {e}")

for i, process in enumerate(processes):
    process.wait()
    print(f"Program {i + 1} finished with exit code {process.returncode}")