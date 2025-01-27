"""
This script runs all the things needed in a single command.
"""
import subprocess

scripts = ["exp_vandelpol_time_mem.py", "exp_laublumi_time_mem.py",
           "table_depz_van_exp.py", "table_depz_laub_exp.py",
           "parallel_table_nmpz_both_exp.py"]

for script in scripts:
    result = subprocess.run(["python3", script])
    
    if result.returncode != 0:
        print(f"Error occurred while running {script}")
        break