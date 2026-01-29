## To Run Baseline:

1. cd ./DeepJoin
2. To run on Freyja Benchmark:
    ** change run_deepjoin_slurm.py to load in run_deepjoin_freyja.sh
    
    ```python run_deepjoin_slurm.py```

3. To run on AutoFJ Benchmark:
    
    ```python run_deepjoin_slurm.py```

    For ENtity linking:


    ```# Sequential execution
        python run_baseline_deepjoin_autofj.py sequential

        # Submit SLURM jobs
        python run_baseline_deepjoin_autofj.py slurm

        # Combine results
        python run_baseline_deepjoin_autofj.py combine

        # Full pipeline (submit + wait + combine)
        python run_baseline_deepjoin_autofj.py pipeline```



## To run our method:
1. python run_offline_embeddings_parallel.py
** Make sure to change paths in main function!

2. python run_offline_sketch_parallel.py
** Make sure to change paths in main function!

3. python run_queries_slurm.py
** Make sure to change paths in main function!
-- if for entity linking experiments, save to entity-linking-experiments/

4. Merge query results: run monitor_and_combine.sh
-- make sure to change the directory path, and # jobs: NUM_JOBS=${2:-10}

5. Evaluate search results: python evaluate_search_results.py


## To run our LLM variation:
1. python run_llm_processing.py
2. python regenerate_pruned_results.py

4. Evaluation:
# Run all benchmarks
./run_evaluation.sh all

# Run specific benchmark
./run_evaluation.sh gdc      # GDC only
./run_evaluation.sh autofj   # AutoFJ with DeepJoin comparison
./run_evaluation.sh autofj-gdc   # AutoFJ+GDC benchmark