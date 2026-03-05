#!/bin/bash
# Run SemSketch evaluation
# Usage: ./run_evaluation.sh [combined|autofj|freyja|wt|gdc]

set -e
cd "$(dirname "$0")"

echo "=============================================="
echo "SemSketch Evaluation"
echo "=============================================="

case "${1:-combined}" in
    autofj)
        python evaluate_retrieval.py --combined --experiments autofj --metrics HITS NDCG MRR --k-values 1 3 5 10 20 30 40 50 --ablation
        ;;
    freyja)
        python evaluate_retrieval.py --combined --experiments freyja --metrics Precision Recall NDCG --k-values 1 3 5 10 20 30 40 50 --ablation
        ;;
    wt)
        python evaluate_retrieval.py --combined --experiments wt --metrics HITS NDCG MRR --k-values 1 3 5 10 20 30 40 50
        ;;
    all)
        python evaluate_retrieval.py --combined --experiments autofj freyja wt --metrics HITS Precision Recall NDCG --k-values 1 3 5 10 20 30 40 50
        ;;
    wdc)
        python evaluate_retrieval.py --llm-annotation  --k-values 1 2 3 4 5 6 7 8 9 10 --save-results results/
        ;;
    *)
        echo "Usage: $0 [autofj|freyja|wt|all|wdc]"
        echo ""
        echo "Modes:"
        echo "  autofj    - AutoFJ only"
        echo "  freyja    - Freyja only"
        echo "  wt        - WT only"
        echo "  all       - All methods"
        echo "  wdc - LLM annotation only"
        exit 1
        ;;
esac
