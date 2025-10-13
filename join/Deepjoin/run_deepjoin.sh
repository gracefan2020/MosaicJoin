#!/usr/bin/env bash
# Make sure to run: conda activate deepjoin38

# Ensure Torch supports safetensors (torch.frombuffer). If not, install CPU-only Torch 2.2.2 and remove conflicting extras.
python - <<'PY'
import sys
try:
    import torch
    ok = hasattr(torch, 'frombuffer')
    print('Torch OK' if ok else 'Torch too old')
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PY
if [ $? -ne 0 ]; then
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet --upgrade "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
  python -m pip uninstall -y torchvision torchaudio 2>/dev/null || true
fi

# Install libs without pulling conflicting torch builds
python -m pip install --quiet --no-deps sentence-transformers
python -m pip install --quiet hnswlib munkres nltk tqdm pandas numpy scikit-learn scipy transformers tokenizers huggingface_hub safetensors

# Ensure NLTK tokenizer data is available (used in preprocessing)
python - <<'PY'
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
PY

python /Users/gracefan/Documents/semantic-join/join/Deepjoin/run_deepjoin.py \
  --dataset_root /Users/gracefan/Documents/LakeBench/datasets/freyja-semantic-join \
  --out_dir /Users/gracefan/Documents/semantic-join/join/Deepjoin/output \
  --model_dir sentence-transformers/all-mpnet-base-v2 \
  --lake_subdir datalake \
  --queries_subdir datalake \
  --split_lake 8 \
  --split_queries 4 \
  --scale 1.0 \
  --K 50 \
  --N 20 \
  --threshold 0.7 \
  --min_matches 1 \
  --queries_csv /Users/gracefan/Documents/semantic-join/datasets/freyja-semantic-join/freyja_query_columns.csv \
  --queries_csv_col target_ds \
  --ground_truth /Users/gracefan/Documents/LakeBench/datasets/freyja-semantic-join/freyja_ground_truth.csv \
  --gt_query_col target_ds \
  --gt_candidate_col candidate_ds

#   --exact_matching \
#   --self_join \