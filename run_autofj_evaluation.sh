# AutoFJ-style: fix precision targets and report the best recall you can get
# (maximize recall s.t. precision >= target), using retrieval-evidence matches.
time python evaluate_autofj_experiment.py \
    --semantic-results autofj_query_results_k1024_t0.1_top10_slurm/all_query_results.csv \
    --output-dir autofj_evaluation_results \
    --use-contributing-entities \
    --precision-targets 0.7 0.8 0.9

#     --skip-entity-linking # only joinability