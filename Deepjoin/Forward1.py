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
    # Early exit if embeddings already exist
    if os.path.exists(storefilename):
        print("Using existing embeddings:", storefilename)
        return storefilename

    # Force CPU device to avoid Torch MPS checks on older Torch versions (e.g., 1.9.0)
    # If local model path is incomplete (e.g., git-lfs not pulled), fall back to HF model name
    try:
        model = SentenceTransformer(model_name, device='cpu')
    except Exception as e:
        try:
            fallback_name = "sentence-transformers/all-mpnet-base-v2"
            print("[WARN] Failed to load local model, falling back to:", fallback_name)
            model = SentenceTransformer(fallback_name, device='cpu')
        except Exception as e2:
            raise e2
    storedata = []
    if os.path.isfile(dataset_file):
        print("Generating embeddings from:", dataset_file)
        try:
            #记载数据
            with open(dataset_file,"rb") as f:
                data = pickle.load(f)
            for ele in tqdm(data):
                key,value = ele
                sentence_embeddings = model.encode(value)
                sentence_embeddings_np = np.array(sentence_embeddings)
                tu1 = (key,sentence_embeddings_np)
                storedata.append(tu1)
        except Exception as e:
            print(e)
    # storefilename already set above
    with open(storefilename,"wb") as f:
        pickle.dump(storedata,f)
    print("data process sucess", storefilename)
    return storefilename



    






