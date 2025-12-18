import numpy as np
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.nn.parallel import DataParallel
import logging
from datetime import datetime
import os
import csv
import pickle
import multiprocessing


def process_onedataset(dataset_file, model_name='output/deepjoin_webtable_training-all-mpnet-base-v2-2023-10-18_19-54-27',
                    storepath="/home/lijiajun/deepjoin/webtable/final_result/", output_filename: str = None):

    path, filename_dataset = os.path.split(dataset_file)
    # If a custom output filename is provided, use it instead of mirroring the input filename
    if output_filename is not None:
        filename_dataset = output_filename

    os.makedirs(storepath,exist_ok=True)
    storefilename = os.path.join(storepath, filename_dataset)
    # Early exit if embeddings already exist (checkpointing)
    if os.path.exists(storefilename):
        print(f"Using existing embeddings: {storefilename}")
        return storefilename

    # Use CUDA if available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        try:
            fallback_name = "sentence-transformers/all-mpnet-base-v2"
            print("[WARN] Failed to load local model, falling back to:", fallback_name)
            model = SentenceTransformer(fallback_name, device=device)
        except Exception as e2:
            raise e2
    
    # Ensure model is on the correct device
    if device == 'cuda':
        model = model.to(device)
        print(f"Model moved to {device}")
        # Check if FP16 is supported (most modern GPUs support it)
        use_fp16 = torch.cuda.is_available()
        if use_fp16:
            print("FP16 mixed precision will be used for faster inference (2x speedup)")
    
    storedata = []
    if os.path.isfile(dataset_file):
        print("Generating embeddings from:", dataset_file)
        try:
            #记载数据
            with open(dataset_file,"rb") as f:
                data = pickle.load(f)
            for ele in tqdm(data):
                key,value = ele
                # Use batch_size to ensure GPU utilization
                # Convert single string to list if needed
                if isinstance(value, str):
                    value = [value]
                # Use larger batch size (128) for better GPU utilization
                # This processes more sentences at once, keeping GPU busy
                # Enable FP16 mixed precision for faster inference if on GPU
                use_fp16 = device == 'cuda' and torch.cuda.is_available()
                if use_fp16:
                    # Use torch.cuda.amp for FP16 mixed precision (2x speedup)
                    with torch.cuda.amp.autocast():
                        sentence_embeddings = model.encode(value, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
                else:
                    sentence_embeddings = model.encode(value, batch_size=128, show_progress_bar=False, convert_to_numpy=True)
                # sentence_embeddings is already numpy array, no need to convert again
                tu1 = (key, sentence_embeddings)
                storedata.append(tu1)
        except Exception as e:
            print(e)
    # storefilename already set above
    with open(storefilename,"wb") as f:
        pickle.dump(storedata,f)
    print("data process sucess", storefilename)
    return storefilename



    






