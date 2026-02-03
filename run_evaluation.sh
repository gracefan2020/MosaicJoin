#!/bin/bash
# Run evaluation on benchmarks
# Usage: ./run_evaluation.sh [gdc|autofj|autofj-gdc|autofj-santos|all]

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
    DEEPJOIN_BASE="autofj-experiments/deepjoin-base-autofj-full-ranked.csv"
    DEEPJOIN_FT="autofj-experiments/deepjoin_ft_autofj.csv"
    GROUND_TRUTH="datasets/autofj_join_benchmark/groundtruth-joinable.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    # Build baseline arguments
    BASELINES=""
    BASELINE_NAMES=""
    
    if [ -f "$DEEPJOIN_BASE" ]; then
        BASELINES="$DEEPJOIN_BASE"
        BASELINE_NAMES="DeepJoin-Base"
    fi
    
    if [ -f "$DEEPJOIN_FT" ]; then
        if [ -n "$BASELINES" ]; then
            BASELINES="$BASELINES $DEEPJOIN_FT"
            BASELINE_NAMES="$BASELINE_NAMES DeepJoin-FT"
        else
            BASELINES="$DEEPJOIN_FT"
            BASELINE_NAMES="DeepJoin-FT"
        fi
    fi
    
    if [ -n "$BASELINES" ]; then
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --baselines $BASELINES \
            --baseline-names $BASELINE_NAMES \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
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
    DEEPJOIN_BASE="autofj-gdc-experiments/deepjoin-base-autofj-gdc-top50.csv"
    DEEPJOIN_FT="autofj-gdc-experiments/deepjoin_ft_autofj_gdc.csv"
    GROUND_TRUTH="datasets/autofj-gdc/groundtruth-joinable.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    # Build baseline arguments
    BASELINES=""
    BASELINE_NAMES=""
    
    if [ -f "$DEEPJOIN_BASE" ]; then
        BASELINES="$DEEPJOIN_BASE"
        BASELINE_NAMES="DeepJoin-Base"
    fi
    
    if [ -f "$DEEPJOIN_FT" ]; then
        if [ -n "$BASELINES" ]; then
            BASELINES="$BASELINES $DEEPJOIN_FT"
            BASELINE_NAMES="$BASELINE_NAMES DeepJoin-FT"
        else
            BASELINES="$DEEPJOIN_FT"
            BASELINE_NAMES="DeepJoin-FT"
        fi
    fi
    
    if [ -n "$BASELINES" ]; then
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --baselines $BASELINES \
            --baseline-names $BASELINE_NAMES \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10
    fi
}

# AutoFJ-Santos Benchmark (Table-Level)
run_autofj_santos() {
    echo "📊 Evaluating AutoFJ-Santos Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-santos-experiments/autofj-santos_query_results_k1024_t0.1_top50_slurm/all_query_results.csv"
    DEEPJOIN_BASE="autofj-santos-experiments/deepjoin-base-autofj-santos.csv"
    DEEPJOIN_FT="autofj-santos-experiments/deepjoin_ft_autofj_santos.csv"
    GROUND_TRUTH="datasets/autofj-santos-small/groundtruth-joinable.csv"
    
    if [ ! -f "$SEMSKETCH_RESULTS" ]; then
        echo "❌ SemSketch results not found: $SEMSKETCH_RESULTS"
        return 1
    fi
    
    # Build baseline arguments
    BASELINES=""
    BASELINE_NAMES=""
    
    if [ -f "$DEEPJOIN_BASE" ]; then
        BASELINES="$DEEPJOIN_BASE"
        BASELINE_NAMES="DeepJoin-Base"
    fi
    
    if [ -f "$DEEPJOIN_FT" ]; then
        if [ -n "$BASELINES" ]; then
            BASELINES="$BASELINES $DEEPJOIN_FT"
            BASELINE_NAMES="$BASELINE_NAMES DeepJoin-FT"
        else
            BASELINES="$DEEPJOIN_FT"
            BASELINE_NAMES="DeepJoin-FT"
        fi
    fi
    
    if [ -n "$BASELINES" ]; then
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --baselines $BASELINES \
            --baseline-names $BASELINE_NAMES \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --level table \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
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
    autofj-santos)
        run_autofj_santos
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
        echo ""
        echo "=============================================="
        echo ""
        run_autofj_santos
        ;;
    *)
        echo "Usage: $0 [gdc|autofj|autofj-gdc|autofj-santos|all]"
        echo ""
        echo "Benchmarks:"
        echo "  gdc          - GDC column-level benchmark"
        echo "  autofj       - AutoFJ table-level benchmark (with DeepJoin-Base & DeepJoin-FT comparison)"
        echo "  autofj-gdc   - AutoFJ-GDC merged table-level benchmark (with DeepJoin-Base & DeepJoin-FT comparison)"
        echo "  autofj-santos - AutoFJ-Santos table-level benchmark (with DeepJoin-Base & DeepJoin-FT comparison)"
        echo "  all          - Run all benchmarks"
        exit 1
        ;;
esac

echo ""
echo "✅ Evaluation complete!"
