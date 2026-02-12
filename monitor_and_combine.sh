#!/bin/bash
# Monitor Slurm jobs and combine results when complete

# OUTPUT_DIR=${1:-"freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm"}
# OUTPUT_DIR=${1:-"autofj-experiments/autofj_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-experiments/gdc_query_results_embeddinggemma_D64_Q64_chamfer_top50_slurm"}
# OUTPUT_DIR=${1:-"autofj-gdc-experiments/autofj-gdc_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-autofj-experiments/gdc-autofj_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-freyja-experiments/gdc-freyja_query_results_k1024_t0.1_top10_slurm"}
# OUTPUT_DIR=${1:-"wt-autofj-experiments/wt-autofj_query_results_k1024_t0.1_top50_slurm_no_column_names"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_k1024_t0.1_greedy_match_top50_slurm_no_column_names"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_k1024_t0.1_chamfer_top50_slurm_no_column_names"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D1024_Q1024_chamfer_top50"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D128_Q1024_chamfer_top50"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D64_Q1024_chamfer_top50"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D128_Q128_chamfer_top50"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D128_Q64_chamfer_top50"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D128_Q32_chamfer_top50"}
# OUTPUT_DIR=${1:-"wikitable-experiments/wikitable_query_results_embeddinggemma_D128_Q128_chamfer_top50"}
# OUTPUT_DIR=${1:-"opendata-experiments/opendata_query_results_embeddinggemma_D128_Q128_chamfer_top50"}

OUTPUT_DIR=${1:-"autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm"}

NUM_JOBS=${2:-20}

echo "🔍 Monitoring Slurm jobs for query processing..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Number of jobs: $NUM_JOBS"
echo ""

echo ""
echo "⏱️  Calculating runtime statistics..."

# Parse per-query timing from slurm output files
AVG_TOTAL=""
AVG_EMBED=""
AVG_SEARCH=""
for outfile in "$OUTPUT_DIR"/slurm_*.out; do
    if [ -f "$outfile" ]; then
        # Extract "Average time per query (embedding + search): 56.3879s"
        val=$(grep -oP "Average time per query \(embedding \+ search\): \K[0-9]+\.?[0-9]*" "$outfile" 2>/dev/null)
        [ -n "$val" ] && AVG_TOTAL="$AVG_TOTAL$val"$'\n'
        # Extract "  - Avg embedding time per query: 1.0246s"
        val=$(grep -oP "Avg embedding time per query: \K[0-9]+\.?[0-9]*" "$outfile" 2>/dev/null)
        [ -n "$val" ] && AVG_EMBED="$AVG_EMBED$val"$'\n'
        # Extract "  - Avg search time per query: 55.3633s"
        val=$(grep -oP "Avg search time per query: \K[0-9]+\.?[0-9]*" "$outfile" 2>/dev/null)
        [ -n "$val" ] && AVG_SEARCH="$AVG_SEARCH$val"$'\n'
    fi
done

# Calculate averages across all jobs
calc_avg() {
    echo "$1" | grep -v "^$" | awk '{ sum += $1; count++ } END { if (count > 0) printf "%.4f", sum/count }'
}

avg_total=$(calc_avg "$AVG_TOTAL")
avg_embed=$(calc_avg "$AVG_EMBED")
avg_search=$(calc_avg "$AVG_SEARCH")

if [ -n "$avg_total" ]; then
    echo "   Avg time per query (embedding + search): ${avg_total}s"
    echo "   Avg embedding time per query: ${avg_embed}s"
    echo "   Avg search time per query: ${avg_search}s"
else
    echo "   No slurm output files found with timing data in $OUTPUT_DIR"
fi

echo ""
echo "🔗 Combining results from all jobs..."
python combine_slurm_results.py "$OUTPUT_DIR" --num-jobs $NUM_JOBS


# echo "🗑️  Removing job_* folders..."
# find "$OUTPUT_DIR" -type d -name "job_*" -exec rm -rf {} +

echo ""
ERROR_COUNT=$(grep -l "Error\|Error\|FAILED" $OUTPUT_DIR/slurm_*.err 2>/dev/null | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Warning: $ERROR_COUNT job(s) had errors. Check the .err files."
else
    echo "✅ No errors found in job outputs"
fi

echo ""
echo "✅ Done! Results are in $OUTPUT_DIR/all_query_results.csv"

