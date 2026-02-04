#!/bin/bash
# Monitor Slurm jobs and combine results when complete

OUTPUT_DIR=${1:-"freyja-experiments/freyja_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"autofj-experiments/autofj_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-experiments/gdc_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"autofj-gdc-experiments/autofj-gdc_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-autofj-experiments/gdc-autofj_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-freyja-experiments/gdc-freyja_query_results_k1024_t0.1_top10_slurm"}
# OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"wt-autofj-experiments/wt-autofj_query_results_k1024_t0.1_top50_slurm_no_column_names"}


# OUTPUT_DIR=${1:-"autofj-santos-experiments/autofj-santos_query_results_k1024_t0.1_top50_slurm"}

NUM_JOBS=${2:-20}

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
echo "📂 Organizing contributing entities files..."

CONTRIB_DIR="$OUTPUT_DIR/contributing_entities"
mkdir -p "$CONTRIB_DIR"

# Move all query_*_contributing_entities.csv into the new folder,
# from any subdirectory inside OUTPUT_DIR
find "$OUTPUT_DIR"/job_* -type f -name "query_*_contributing_entities.csv" -exec mv {} "$CONTRIB_DIR"/ \;

echo "🗑️  Removing job_* folders..."
find "$OUTPUT_DIR" -type d -name "job_*" -exec rm -rf {} +

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
echo "   All contributing_entities have been moved to $CONTRIB_DIR"

