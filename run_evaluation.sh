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
        python evaluate_retrieval.py --combined --experiments autofj --metrics HITS NDCG MRR --k-values 1 3 5 10 20 30 40 50
        ;;
    freyja)
        python evaluate_retrieval.py --combined --experiments freyja --metrics HITS Precision Recall NDCG MRR --k-values 1 3 5 10 20 30 40 50
        ;;
    wt)
        python evaluate_retrieval.py --combined --experiments wt --metrics HITS NDCG MRR --k-values 1 3 5 10 20 30 40 50
        ;;
    *)
        echo "Usage: $0 [combined|autofj|freyja|wt]"
        echo ""
        echo "Modes:"
        echo "  autofj    - AutoFJ only"
        echo "  freyja    - Freyja only"
        echo "  wt        - WT only"
        exit 1
        ;;
esac

echo ""
echo "✅ Evaluation complete!"
