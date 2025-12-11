#!/bin/bash
# Monitor Slurm jobs and combine results when complete

# OUTPUT_DIR=${1:-"query_results_k1024_t0.7_top50_deepjoin_N100_K500_T0.6_slurm"}
OUTPUT_DIR=${1:-"query_results_k1024_t0.1_top50_semantic_matches_slurm"}
NUM_JOBS=${2:-5}

echo "🔍 Monitoring Slurm jobs for query processing..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Number of jobs: $NUM_JOBS"
echo ""

# Check job status
while true; do
    RUNNING=$(squeue -u $USER -n semantic_query --format="%.10i" --noheader 2>/dev/null | wc -l)
    
    if [ "$RUNNING" -eq 0 ]; then
        echo "✅ All jobs completed!"
        break
    else
        echo "⏳ Jobs still running: $RUNNING"
        sleep 10
    fi
done

echo ""
echo "🔗 Combining results from all jobs..."
python combine_slurm_results.py "$OUTPUT_DIR" --num-jobs $NUM_JOBS

echo ""
echo "📊 Checking for errors in job outputs..."
ERROR_COUNT=$(grep -l "Error\|Error\|FAILED" $OUTPUT_DIR/slurm_*.err 2>/dev/null | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Warning: $ERROR_COUNT job(s) had errors. Check the .err files."
else
    echo "✅ No errors found in job outputs"
fi

echo ""
echo "✅ Done! Results are in $OUTPUT_DIR/all_query_results.csv"

