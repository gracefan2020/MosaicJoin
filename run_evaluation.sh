#!/bin/bash
# Run evaluation on benchmarks
# Usage: ./run_evaluation.sh [gdc|autofj|autofj-gdc|all]

set -e

BENCHMARK=${1:-all}

echo "=============================================="
echo "SemSketch Evaluation Script"
echo "=============================================="
echo ""

# GDC Benchmark (Column-Level)
run_gdc() {
    echo "📊 Evaluating GDC Benchmark (Column-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="gdc-experiments/gdc_query_results_k1024_t0.1_top10_slurm/all_query_results.csv"
    GROUND_TRUTH="datasets/gdc/join_col_groundtruth.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    python evaluate_retrieval.py \
        --results "$SEMSKETCH_RESULTS" \
        --ground-truth "$GROUND_TRUTH" \
        --level column \
        --name "SemSketch" \
        --k-values 1 3 5 10
}

# AutoFJ Benchmark (Table-Level)
run_autofj() {
    echo "📊 Evaluating AutoFJ Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-experiments/autofj_query_results_k1024_t0.1_top10_slurm/all_query_results.csv"
    # DEEPJOIN_RESULTS="autofj-experiments/autofj_deepjoin_baseline_k10_n10_t0.1/all_query_results.csv"
    DEEPJOIN_RESULTS="autofj-experiments/deepjoin-autofj-full-ranked.csv"
    GROUND_TRUTH="datasets/autofj_join_benchmark/groundtruth-joinable.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    # Check if DeepJoin baseline exists
    if [ -f "$DEEPJOIN_RESULTS" ]; then
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --baseline "$DEEPJOIN_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --baseline-name "DeepJoin" \
            --k-values 1 3 5 10
    else
        echo "⚠️  DeepJoin baseline not found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10
    fi
}

# AutoFJ-GDC Merged Benchmark (Table-Level)
run_autofj_gdc() {
    echo "📊 Evaluating AutoFJ-GDC Merged Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-gdc-experiments/autofj-gdc_query_results_k1024_t0.1_top10_slurm/all_query_results.csv"
    DEEPJOIN_RESULTS="autofj-gdc-experiments/deepjoin-autofj-gdc-top50.csv"
    GROUND_TRUTH="datasets/autofj-gdc/groundtruth-joinable.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    # Check if DeepJoin baseline exists
    if [ -f "$DEEPJOIN_RESULTS" ]; then
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --baseline "$DEEPJOIN_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --baseline-name "DeepJoin" \
            --k-values 1 3 5 10
    else
        echo "⚠️  DeepJoin baseline not found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10
    fi
}

# Run based on argument
case $BENCHMARK in
    gdc)
        run_gdc
        ;;
    autofj)
        run_autofj
        ;;
    autofj-gdc)
        run_autofj_gdc
        ;;
    all)
        run_gdc
        echo ""
        echo "=============================================="
        echo ""
        run_autofj
        echo ""
        echo "=============================================="
        echo ""
        run_autofj_gdc
        ;;
    *)
        echo "Usage: $0 [gdc|autofj|autofj-gdc|all]"
        echo ""
        echo "Benchmarks:"
        echo "  gdc       - GDC column-level benchmark"
        echo "  autofj    - AutoFJ table-level benchmark (with DeepJoin comparison)"
        echo "  autofj-gdc - AutoFJ-GDC merged table-level benchmark (with DeepJoin comparison)"
        echo "  all       - Run all benchmarks"
        exit 1
        ;;
esac

echo ""
echo "✅ Evaluation complete!"
