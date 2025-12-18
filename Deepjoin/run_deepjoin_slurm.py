import os
import subprocess

# Configuration for AutoFuzzyJoin dataset
# Use two-stage submission: preprocessing (CPU) -> embedding (GPU)
USE_TWO_STAGE = True  # Set to False to use single GPU job (old method)

if USE_TWO_STAGE:
    print("=" * 60)
    print("TWO-STAGE SUBMISSION MODE")
    print("=" * 60)
    
    # Stage 1: Preprocessing (CPU-only, no GPU needed)
    print("\nStage 1: Submitting preprocessing job (CPU-only)...")
    preprocess_cmd = [
        'sbatch',
        '--nodes=1',
        '--tasks-per-node=1',
        '--cpus-per-task=8',
        '--mem=32GB',
        '--time=2:00:00',
        '--output=deepjoin_slurm_autofj_preprocess.log',
        'run_deepjoin_autofj_preprocess.sh'
    ]
    
    print(f"Running: {' '.join(preprocess_cmd)}")
    result1 = subprocess.run(preprocess_cmd, capture_output=True, text=True)
    
    if result1.returncode != 0:
        print(f"ERROR: Preprocessing job submission failed")
        print(result1.stderr)
        exit(1)
    
    # Extract job ID from output (format: "Submitted batch job 12345")
    preprocess_output = result1.stdout.strip()
    print(f"Preprocessing job output: {preprocess_output}")
    
    try:
        preprocess_job_id = preprocess_output.split()[-1]
        print(f"Preprocessing job ID: {preprocess_job_id}")
    except:
        print("WARNING: Could not parse preprocessing job ID. Continuing anyway...")
        preprocess_job_id = None
    
    # Stage 2: Embedding (GPU, depends on preprocessing)
    print("\nStage 2: Submitting embedding job (GPU)...")
    embed_cmd = [
        'sbatch',
        '--gres=gpu:1',
        '--nodes=1',
        '--tasks-per-node=1',
        '--cpus-per-task=8',
        '--mem=32GB',
        '--time=2:00:00',
        '--output=deepjoin_slurm_autofj_embed.log',
    ]
    
    # Add dependency if we got a job ID
    if preprocess_job_id:
        embed_cmd.extend(['--dependency=afterok:' + preprocess_job_id])
        print(f"Embedding job will wait for preprocessing job {preprocess_job_id} to complete")
    
    embed_cmd.append('run_deepjoin_autofj_embed.sh')
    
    print(f"Running: {' '.join(embed_cmd)}")
    result2 = subprocess.run(embed_cmd, capture_output=True, text=True)
    
    if result2.returncode != 0:
        print(f"ERROR: Embedding job submission failed")
        print(result2.stderr)
        exit(1)
    
    embed_output = result2.stdout.strip()
    print(f"Embedding job output: {embed_output}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Both jobs submitted!")
    print("=" * 60)
    print(f"Preprocessing job: {preprocess_output}")
    print(f"Embedding job: {embed_output}")
    print("\nMonitor jobs with: squeue -u $USER")
    
else:
    # Single-stage mode (old method - GPU job does everything)
    print("=" * 60)
    print("SINGLE-STAGE SUBMISSION MODE (GPU job)")
    print("=" * 60)
    
    slurm_cmd = 'sbatch --gres=gpu:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=8 --mem=32GB --time=4:00:00 --output=deepjoin_slurm_autofj_k10_n10_t0.1.log run_deepjoin_autofj.sh'
    
    print(f"Running slurm command: {slurm_cmd}")
    
    result = os.system(slurm_cmd)
    
    if result != 0:
        print(f"ERROR: SLURM submission failed")
    else:
        print(f"Successfully submitted")