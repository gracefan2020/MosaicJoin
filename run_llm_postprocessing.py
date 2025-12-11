import os
import sys

# Default query results directory
QUERY_RESULTS_DIR = "query_results_k1024_t0.1_top50_slurm"

# Build the command to run
python_cmd = f'python llm_postprocessing.py "{QUERY_RESULTS_DIR}"'

# Build the sbatch command with all SLURM directives
slurm_cmd = (
    f'sbatch '
    f'--job-name=llm_postprocessing '
    f'--error=llm_postprocessing.err '
    f'--output=/dev/null '
    f'--time=24:00:00 '
    f'--cpus-per-task=4 '
    f'--mem=32G '
    f'--gres=gpu:1 '
    f'--wrap="echo \\"Starting LLM postprocessing for: {QUERY_RESULTS_DIR}\\"; '
    f'echo \\"Start time: $(date)\\"; '
    f'{python_cmd}; '
    f'echo \\"End time: $(date)\\"; '
    f'echo \\"Job completed\\""'
)

print(f"Submitting SLURM job for: {QUERY_RESULTS_DIR}")
print(f"SLURM command: {slurm_cmd}")
    
result = os.system(slurm_cmd)

if result != 0:
    print(f"ERROR: SLURM submission failed")
    sys.exit(1)
else:
    print(f"Successfully submitted job")