#!/bin/bash
# Run evaluation on benchmarks
# Usage: ./run_evaluation.sh [freyja|gdc|gdc-autofj|gdc-freyja|autofj|autofj-wdc|autofj-santos|wt|wt-autofj|wikitable|opendata|all]

set -e

BENCHMARK=${1:-all}

echo "=============================================="
echo "SemSketch Evaluation Script"
echo "=============================================="
echo ""

# Freyja Benchmark (Column-Level)
run_freyja() {
    echo "📊 Evaluating Freyja Benchmark (Column-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="freyja-experiments/freyja_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv"
    # DEEPJOIN_BASE="freyja-experiments/deepjoin-base-freyja.csv"
    DEEPJOIN_FT="freyja-experiments/deepjoin_ft_freyja.csv"
    GROUND_TRUTH="datasets/freyja-semantic-join/freyja_ground_truth_no_column_names.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# GDC Benchmark (Column-Level)
run_gdc() {
    echo "📊 Evaluating GDC Benchmark (Column-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="gdc-experiments/gdc_query_results_embeddinggemma_D64_Q64_chamfer_top50_slurm/all_query_results.csv"
    # DEEPJOIN_BASE="gdc-experiments/deepjoin-base-gdc.csv"
    DEEPJOIN_FT="gdc-experiments/deepjoin_ft_gdc.csv"
    GROUND_TRUTH="datasets/gdc-breakdown/join_col_groundtruth.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# AutoFJ Benchmark (Table-Level)
run_autofj() {
    echo "📊 Evaluating AutoFJ Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-experiments/autofj_query_results_embeddinggemma_D128_Q128_chamfer_top50_slurm/all_query_results.csv"
    # DEEPJOIN_BASE="autofj-experiments/deepjoin-base-autofj-full-ranked.csv"
    DEEPJOIN_FT="autofj-experiments/deepjoin_ft_autofj_grace.csv"
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# autofj-wdc Merged Benchmark (Table-Level)
run_autofj_wdc() {
    echo "📊 Evaluating AutoFJ-WDC Merged Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-wdc-experiments/autofj-wdc_query_results_embeddinggemma_D128_Q0_chamfer_top50_slurm/all_query_results.csv"
    DEEPJOIN_FT="autofj-wdc-experiments/deepjoin_ft_autofj-wdc.csv"
    GROUND_TRUTH="datasets/autofj-wdc/groundtruth-joinable.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# GDC-AutoFJ Benchmark (Column-Level)
run_gdc_autofj() {
    echo "📊 Evaluating GDC-AutoFJ Benchmark (Column-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="gdc-autofj-experiments/gdc-autofj_query_results_k1024_t0.1_top50_slurm/all_query_results.csv"
    # DEEPJOIN_BASE="gdc-autofj-experiments/deepjoin-base-gdc-autofj.csv"
    DEEPJOIN_FT="gdc-autofj-experiments/deepjoin_ft_gdc-autofj.csv"
    GROUND_TRUTH="datasets/gdc-autofj/join_col_groundtruth.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# GDC-Freyja Benchmark (Column-Level)
run_gdc_freyja() {
    echo "📊 Evaluating GDC-Freyja Benchmark (Column-Level)..."
    echo ""
    
    # SEMSKETCH_RESULTS="gdc-freyja-experiments/gdc-freyja_query_results_k1024_t0.1_top10_slurm/all_query_results.csv"
    SEMSKETCH_RESULTS="gdc-autofj-experiments/gdc-autofj_query_results_k1024_t0.1_top10_slurm/all_query_results.csv" # TEMP: REMOVE ONCE WE HAVE FREYJA RESULTS

    # DEEPJOIN_BASE="gdc-freyja-experiments/deepjoin-base-gdc-freyja.csv"
    DEEPJOIN_FT="gdc-freyja-experiments/deepjoin_ft_gdc-freyja.csv"
    GROUND_TRUTH="datasets/gdc-freyja/join_col_groundtruth.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10
    fi
}

# WT Benchmark (Column-Level)
run_wt() {
    echo "📊 Evaluating WT Benchmark (Column-Level)..."
    echo ""
    
    # SEMSKETCH_RESULTS="wt-experiments/wt_query_results_k1024_t0.1_chamfer_top50_slurm_no_column_names/all_query_results.csv"
    SEMSKETCH_RESULTS="wt-experiments/wt_query_results_embeddinggemma_D128_Q128_chamfer_top50/all_query_results.csv"
    # DEEPJOIN_BASE="wt-experiments/deepjoin-base-wt.csv"
    DEEPJOIN_FT="wt-experiments/deepjoin_ft_wt_no_col_names.csv"
    GROUND_TRUTH="datasets/wt/wt_join_groundtruth_no_column_names.csv"
    echo "SEMSKETCH_RESULTS: $SEMSKETCH_RESULTS"
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}


# WT-AutoFJ Benchmark (Column-Level)
run_wt_autofj() {
    echo "📊 Evaluating WT-AutoFJ Benchmark (Column-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="wt-autofj-experiments/wt-autofj_query_results_k1024_t0.1_top50_slurm_no_column_names/all_query_results.csv"
    # DEEPJOIN_BASE="wt-autofj-experiments/deepjoin-base-wt-autofj.csv"
    DEEPJOIN_FT="wt-autofj-experiments/deepjoin_ft_wt-autofj_no_col_names.csv"
    GROUND_TRUTH="datasets/wt-autofj/wt_join_groundtruth_no_column_names.csv"
    
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# WikiTable Benchmark (Table-Level) - Snoopy
run_wikitable() {
    echo "📊 Evaluating WikiTable Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="wikitable-experiments/wikitable_query_results_embeddinggemma_D128_Q128_chamfer_top50/all_query_results.csv"
    DEEPJOIN_BASE="wikitable-experiments/deepjoin_base_wikitable.csv"
    # DEEPJOIN_FT="wikitable-experiments/deepjoin_ft_wikitable.csv"
    GROUND_TRUTH="datasets/WikiTable/groundtruth-joinable.csv"
    
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
            --name "SemSketch" \
            --k-values 5 15 25
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 5 15 25
    fi
}

# opendata Benchmark (Table-Level) - Snoopy
run_opendata() {
    echo "📊 Evaluating opendata Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="opendata-experiments/opendata_query_results_embeddinggemma_D128_Q128_chamfer_top50/all_query_results.csv"
    # DEEPJOIN_FT="opendata-experiments/deepjoin_ft_opendata.csv"
    GROUND_TRUTH="datasets/opendata/groundtruth-joinable.csv"
    
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
            --name "SemSketch" \
            --k-values 5 15 25
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 5 15 25
    fi
}

# AutoFJ-Santos Benchmark (Table-Level)
run_autofj_santos() {
    echo "📊 Evaluating AutoFJ-Santos Benchmark (Table-Level)..."
    echo ""
    
    SEMSKETCH_RESULTS="autofj-santos-experiments/autofj-santos_query_results_k1024_t0.1_top50_slurm/all_query_results.csv"
    # DEEPJOIN_BASE="autofj-santos-experiments/deepjoin-base-autofj-santos.csv"
    DEEPJOIN_FT="autofj-santos-experiments/deepjoin_ft_autofj-santos.csv"
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
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    else
        echo "⚠️  No baselines found, evaluating SemSketch only"
        python evaluate_retrieval.py \
            --results "$SEMSKETCH_RESULTS" \
            --ground-truth "$GROUND_TRUTH" \
            --name "SemSketch" \
            --k-values 1 3 5 10 20 30 40 50
    fi
}

# Run based on argument
case $BENCHMARK in
    freyja)
        run_freyja
        ;;
    gdc)
        run_gdc
        ;;
    gdc-autofj)
        run_gdc_autofj
        ;;
    gdc-freyja)
        run_gdc_freyja
        ;;
    autofj)
        run_autofj
        ;;
    autofj-wdc)
        run_autofj_wdc
        ;;
    autofj-santos)
        run_autofj_santos
        ;;
    wt)
        run_wt
        ;;
    wt-autofj)
        run_wt_autofj
        ;;
    wikitable)
        run_wikitable
        ;;
    opendata)
        run_opendata
        ;;
    all)
        run_freyja
        echo ""
        echo "=============================================="
        echo ""
        run_gdc
        echo ""
        echo "=============================================="
        echo ""
        run_gdc_autofj
        echo ""
        echo "=============================================="
        echo ""
        run_gdc_freyja
        echo ""
        echo "=============================================="
        echo ""
        run_autofj
        echo ""
        echo "=============================================="
        echo ""
        run_autofj_wdc
        echo ""
        echo "=============================================="
        echo ""
        run_autofj_santos
        echo ""
        echo "=============================================="
        echo ""
        run_wt
        echo ""
        echo "=============================================="
        echo ""
        run_wt_autofj
        echo ""
        echo "=============================================="
        echo ""
        run_wikitable
        echo ""
        echo "=============================================="
        echo ""
        run_opendata
        ;;
    *)
        echo "Usage: $0 [freyja|gdc|gdc-autofj|gdc-freyja|autofj|autofj-wdc|autofj-santos|wt|wt-autofj|wikitable|opendata|all]"
        echo ""
        echo "Benchmarks:"
        echo "  freyja        - Freyja column-level benchmark (with DeepJoin-FT comparison)"
        echo "  gdc           - GDC column-level benchmark (with DeepJoin-FT comparison)"
        echo "  gdc-autofj    - GDC-AutoFJ column-level benchmark (with DeepJoin-FT comparison)"
        echo "  gdc-freyja    - GDC-Freyja column-level benchmark (with DeepJoin-FT comparison)"
        echo "  autofj        - AutoFJ table-level benchmark (with DeepJoin-Base & DeepJoin-FT comparison)"
        echo "  autofj-wdc    - autofj-wdc merged table-level benchmark (with DeepJoin-FT comparison)"
        echo "  autofj-santos - AutoFJ-Santos table-level benchmark (with DeepJoin-FT comparison)"
        echo "  wt            - WT column-level benchmark (with DeepJoin-FT comparison)"
        echo "  wt-autofj     - WT-AutoFJ column-level benchmark (with DeepJoin-FT comparison)"
        echo "  wikitable     - WikiTable table-level benchmark (Snoopy, with DeepJoin comparison)"
        echo "  opendata      - opendata table-level benchmark (Snoopy, with DeepJoin comparison)"
        echo "  all           - Run all benchmarks"
        exit 1
        ;;
esac

echo ""
echo "✅ Evaluation complete!"
