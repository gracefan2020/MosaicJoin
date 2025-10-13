#!/usr/bin/env bash
set -euo pipefail

# Configuration variables
: "${GROUND_TRUTH:=datasets/freyja-semantic-join/freyja_ground_truth.csv}"
: "${DEEPJOIN_RESULTS:=join/Deepjoin/output/deepjoin_results_T0.7_exact.csv}"
: "${SEMANTIC_SKETCHES_DIR:=freyja-semantic-join-results-k1024}"
: "${ANALYSES_DIR:=analyses-disagreements-index-k1024}"

echo "Analyzing DeepJoin vs Semantic Sketch disagreements"
echo "  Ground truth          : ${GROUND_TRUTH}"
echo "  DeepJoin results      : ${DEEPJOIN_RESULTS}"
echo "  Semantic sketches dir : ${SEMANTIC_SKETCHES_DIR}"
echo "  Analyses out dir      : ${ANALYSES_DIR}"

python scripts/analyze_disagreements.py \
  --ground-truth "${GROUND_TRUTH}" \
  --deepjoin-results "${DEEPJOIN_RESULTS}" \
  --semantic-sketches-dir "${SEMANTIC_SKETCHES_DIR}" \
  --out-dir "${ANALYSES_DIR}"

echo "Done. Wrote analyses to: ${ANALYSES_DIR}"


