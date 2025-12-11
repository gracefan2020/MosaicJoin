import os

slurm_cmd = f'sbatch --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=4GB --time=4:00:00 --output=deepjoin_slurm_final_k50_t0.1.log run_deepjoin.sh'
        
print(f"Running slurm command: {slurm_cmd}")
    
result = os.system(slurm_cmd)

if result != 0:
    print(f"ERROR: SLURM submission failed")
else:
    print(f"Successfully submitted")