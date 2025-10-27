import os

slurm_cmd = f'sbatch --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=4GB --time=2:00:00 --output=llm_postprocessing_deepjoin_slurm_2.log run_llm_postprocessing.sh'
        
print(f"Running slurm command: {slurm_cmd}")
    
result = os.system(slurm_cmd)

if result != 0:
    print(f"ERROR: SLURM submission failed")
else:
    print(f"Successfully submitted")