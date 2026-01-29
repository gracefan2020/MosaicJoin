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


## To run our LLM variation (deprecated):
1. python run_llm_processing.py
2. python regenerate_pruned_results.py

# Evaluation:
## Run all benchmarks
./run_evaluation.sh all

## Run specific benchmark
./run_evaluation.sh gdc      # GDC only
./run_evaluation.sh autofj   # AutoFJ with DeepJoin comparison
./run_evaluation.sh autofj-gdc   # AutoFJ+GDC benchmark