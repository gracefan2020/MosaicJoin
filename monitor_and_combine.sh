#!/bin/bash
# Monitor Slurm jobs and combine results when complete

# OUTPUT_DIR=${1:-"freyja-experiments/freyja_query_results_k1024_chamfer_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"autofj-experiments/autofj_query_results_k1024_t0.1_top50_slurm"}
# OUTPUT_DIR=${1:-"gdc-experiments/gdc_query_results_k1024_t0.1_top50_slurm"}
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
OUTPUT_DIR=${1:-"wt-experiments/wt_query_results_embeddinggemma_D128_Q32_chamfer_top50"}

# OUTPUT_DIR=${1:-"autofj-santos-experiments/autofj-santos_query_results_k1024_t0.1_top50_slurm"}

NUM_JOBS=${2:-20}

echo "🔍 Monitoring Slurm jobs for query processing..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Number of jobs: $NUM_JOBS"
echo ""

echo ""
echo "⏱️  Calculating runtime statistics..."

# Parse "Total processing time: X.XXs" from slurm output files
RUNTIMES=""
for outfile in "$OUTPUT_DIR"/slurm_*.out; do
    if [ -f "$outfile" ]; then
        # Extract "Total processing time: 32.01s" -> 32.01
        runtime=$(grep -oP "Total processing time: \K[0-9]+\.?[0-9]*" "$outfile" 2>/dev/null)
        if [ -n "$runtime" ]; then
            RUNTIMES="$RUNTIMES$runtime"$'\n'
        fi
    fi
done
RUNTIMES=$(echo "$RUNTIMES" | grep -v "^$")

if [ -n "$RUNTIMES" ]; then
    # Calculate statistics using awk (handles floating point)
    echo "$RUNTIMES" | awk '
    BEGIN { min = 999999; max = 0; sum = 0; count = 0 }
    {
        val = $1 + 0  # Convert to number
        if (val > 0) {
            sum += val
            count++
            if (val < min) min = val
            if (val > max) max = val
        }
    }
    END {
        if (count > 0) {
            avg = sum / count
            printf "   Min runtime:   %.2f seconds (%.2f minutes)\n", min, min/60
            printf "   Max runtime:   %.2f seconds (%.2f minutes)\n", max, max/60
            printf "   Avg runtime:   %.2f seconds (%.2f minutes)\n", avg, avg/60
            printf "   Total runtime: %.2f seconds (%.2f minutes)\n", sum, sum/60
            printf "   Number of jobs: %d\n", count
        } else {
            print "   Could not parse runtime data from output files"
        }
    }'
else
    echo "   No slurm output files found with timing data in $OUTPUT_DIR"
fi

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
ERROR_COUNT=$(grep -l "Error\|Error\|FAILED" $OUTPUT_DIR/slurm_*.err 2>/dev/null | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Warning: $ERROR_COUNT job(s) had errors. Check the .err files."
else
    echo "✅ No errors found in job outputs"
fi

echo ""
echo "✅ Done! Results are in $OUTPUT_DIR/all_query_results.csv"
echo "   All contributing_entities have been moved to $CONTRIB_DIR"

