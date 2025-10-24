import os
import pickle
import random
import shutil
import sys
import nltk
import hashlib
import numpy as np

import pandas as pd
import multiprocessing
from multiprocessing import Process, Queue
from tqdm import  tqdm
import multiprocessing
import time
import torch.multiprocessing

"""
On some platforms (e.g., macOS), setting CPU affinity via os.sched_setaffinity
is not supported. Guard this to avoid import-time crashes.
"""
try:
    cpu_cores = [i for i in range(os.cpu_count() or 1)]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(os.getpid(), cpu_cores)
except Exception:
    # Silently skip affinity configuration if unsupported
    pass



torch.multiprocessing.set_sharing_strategy('file_system')


# Constants for priority sampling
PHI_FRACTION = 0.6180339887  # φ - 1

def fibonacci_hash(x):
    """Calculate fibonacci hash for priority sampling."""
    result = (x * PHI_FRACTION) % 1  # Take fractional part
    return result

def get_samples(values, mode="frequent"):
    """
    Sample values from a pandas Series using different strategies.
    
    Args:
        values: pandas Series containing the values to sample
        mode: sampling strategy ('random', 'frequent', 'mixed', 'weighted', 'priority_sampling')
            - 'random': completely random sampling from unique values
            - 'frequent': only the most frequent values (original Deepjoin behavior)
            - 'mixed': combination of frequent and diverse values
            - 'weighted': weighted sampling based on value counts
            - 'priority_sampling': uses priority sampling based on frequency and hash of the values
    
    Returns:
        List of string representations of sampled values
    """
    unique_values = values.dropna().unique()
    total_unique = len(unique_values)
    
    # Use all unique values (no sampling limit)
    n = total_unique
    
    if mode == "random":
        # Completely random sampling
        random_indices = np.random.choice(total_unique, size=n, replace=False)
        sampled_values = unique_values[random_indices]
        tokens = sorted([str(val) for val in sampled_values])
        
    elif mode == "frequent":
        # Only most frequent values (original Deepjoin behavior)
        value_counts = values.dropna().value_counts()
        tokens = [str(val) for val in value_counts.index.tolist()]
        tokens.sort()
        
    elif mode == "mixed":
        # Mix of most frequent and evenly spaced values
        n_frequent = n // 2
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n_frequent).index.tolist()
        
        # Calculate evenly spaced samples for diversity
        n_diverse = n - n_frequent
        spacing_interval = max(1, total_unique // n_diverse)
        diverse_values = unique_values[::spacing_interval][:n_diverse]
        
        # Combine frequent and diverse samples, remove duplicates
        tokens = sorted(set([str(val) for val in most_frequent_values + list(diverse_values)]))
        
    elif mode == "weighted":
        # Weighted sampling based on value counts
        value_counts = values.dropna().value_counts(sort=False)
        weights = value_counts / value_counts.sum()
        sampled_indices = np.random.choice(
            total_unique, size=n, replace=False, p=weights
        )
        sampled_values = unique_values[sampled_indices]
        tokens = sorted([str(val) for val in sampled_values])
        
    elif mode == "priority_sampling":
        value_counts = values.dropna().value_counts(sort=False)
        
        # Calculate priorities: qi = freq / hash(value)
        priorities = pd.Series({
            val: freq / fibonacci_hash(hash(str(val)) % (2**32))
            for val, freq in value_counts.items()
        })
        
        # Select the top elements based on priority scores
        sampled_values = priorities.nlargest(n).index.tolist()
        tokens = sorted([str(val) for val in sampled_values])
        
    else:
        # Default to frequent sampling
        value_counts = values.dropna().value_counts()
        tokens = [str(val) for val in value_counts.index.tolist()]
        tokens.sort()
    
    return tokens


def analyze_column_values(df, column_name, sampling_mode="frequent"):

    # Get column values and apply sampling strategy
    column_values = df[column_name].astype(str)
    
    # Use the new sampling method
    sampled_values = get_samples(column_values, mode=sampling_mode)
    
    n = len(sampled_values)
    # Join sampled values with commas
    col = ', '.join(sampled_values)
    
    # Calculate statistics for the sampled values
    lengths = [len(str(value)) for value in sampled_values]
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    
    tokens = f"{column_name} contains {str(n)} values ({str(max_len)}, {str(min_len)}, {str(avg_len)}): {col}"
    
    # Tokenize and truncate
    tokens = nltk.word_tokenize(tokens)
    truncated_tokens = tokens[:512]
    truncated_sentence = ' '.join(truncated_tokens)
    return truncated_sentence

# 输入一个df 文件句柄，输出 是一个字符串list 的，每一个字符串表示这一列 的数据的串行
def evaluate4(df, sampling_mode="frequent"):
    columns = df.columns.tolist()
    sentens_list = []
    for column in columns:
        s = analyze_column_values(df, column, sampling_mode=sampling_mode)
        sentens_list.append(s)
    return sentens_list



def get_file_columns(file_path):
    df = pd.read_csv(file_path,engine='python',nrows =1)
    columns = len(df.columns)
    return columns

def partition_files(file_paths, m):
    random.shuffle(file_paths)
    file_info = []  # 存储文件列表及其对应的列数的元组列表

    current_group = []  # 当前文件组
    current_columns = 0  # 当前文件组的列数和
    stime = time.time()
    for file_path in file_paths:
        columns = get_file_columns(file_path)
        # 如果当前文件组的列数和加上当前文件的列数超过m，则将当前文件组加入结果列表，并创建新的文件组
        if current_columns + columns > m:
            file_info.append(current_group)
            current_group = [file_path + "_" + str(columns)]
            current_columns = columns
        else:
            current_group.append(file_path+ "_" + str(columns))
            current_columns += columns
    endtime = time.time()
    #print(f"after partioning : {endtime - stime}")
    # 添加最后一个文件组
    if current_group:
        file_info.append(current_group)
    return file_info



def create_folder(path):
    if os.path.exists(path):
        # 如果路径存在，删除文件夹及其下的所有文件
        shutil.rmtree(path)
        print("Folder and its content deleted.")

    # 创建新的文件夹
    os.makedirs(path)
    print("Folder created.")


def read_pkl_files(folder_path):
    # 获取文件夹中的文件列表
    file_list = os.listdir(folder_path)

    re_dict = {}

    # 遍历文件列表
    for file_name in file_list:
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否为.pkl文件
        if file_name.endswith(".pkl"):
            try:
                # 打开.pkl文件并加载变量
                with open(file_path, 'rb') as file:
                    obj = pickle.load(file)

                re_dict.update(obj)
            except Exception as e:
                print("Error occurred while reading", file_name, ":", str(e))
    return re_dict


def process_task4(i,input_values,queue,queue_inforgather,file_dic_path,sampling_mode="frequent"):

    dict = {}
    for input_value in input_values:
        k = struct_dic_key(input_value)
        try:
            df = pd.read_csv(input_value, low_memory=False)
        except Exception as e:
            print("error filename:",input_value)
            continue
        #embdings = evaluate4(df).detach().numpy()
        embdings = evaluate4(df, sampling_mode=sampling_mode)
        dict[k] = embdings
        queue.put(1)
    filename = os.path.join(file_dic_path,str(i)+".pkl")
    with open(filename, 'wb') as file:
        pickle.dump(dict, file)
    queue.put((-1, "test-pid"))




def struct_dic_key(filepath):
    elelist = filepath.split(os.sep)
    return elelist[-2] + "-" + elelist[-1]

def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result

def process_table_sentense(filepathstore,datadir,data_pkl_name,tmppath= "/data/lijiajun/webtable_tmp",split_num=10,sampling_mode="frequent"):

    # filepathstore = "/data/lijiajun/opendata/large"
    # dir = "/data/opendata/large/query/"
    # list_of_tuples_name = "opendata_large_query_new.pkl"
    # file_dic_path = "/data/lijiajun/webtable_tmp"
    # split_num = 10
    list_of_tuples_name = data_pkl_name
    dir = datadir
    file_dic_path = tmppath

    os.makedirs(filepathstore,exist_ok=True)
    # create_folder(file_dic_path)

    filelist = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            if file == 'small_join.csv' or file  == 'large_join.csv':
                continue
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                #print(filepath)
                filelist.append(filepath)

            else:
                print(f"file: {filepath} is not a file ,pass")
    print(f"split1 all file ,filelistlen: {len(filelist)} added to filelist")


    inputs = filelist

    # 指定包含查询表的文件夹路径

    # 获取文件夹中的所有文件名

    sub_file_ls = split_list(inputs, split_num)
    process_list = []

    #####
    # 为每个进程创建一个队列
    queues = [Queue() for i in range(split_num)]
    # queue = Queue()
    # 一个用于标识所有进程已结束的数组
    finished = [False for i in range(split_num)]

    # 为每个进程创建一个进度条
    bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
    # 用于保存每个进程的返回结果
    results = [None for i in range(split_num)]
    queue_inforgather = multiprocessing.Manager().Queue()

    for i in range(split_num):
        process = Process(target=process_task4, args=(i,sub_file_ls[i], queues[i], queue_inforgather,file_dic_path,sampling_mode))
        process_list.append(process)
        process.start()

    while True:
        for i in range(split_num):
            queue = queues[i]
            bar = bars[i]
            try:
                # 从队列中获取数据
                # 这里需要用非阻塞的get_nowait或get(True)
                # 如果用get()，当某个进程在某一次处理的时候花费较长时间的话，会把后面的进程的进度条阻塞着
                # 一定要try捕捉错误，get_nowait读不到数据时会跑出错误
                res = queue.get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    # 某个进程已经处理完毕
                    finished[i] = True
                    results[i] = res[1]
                    continue
                bar.update(res)
            except Exception as e:
                continue

                # 所有进程处理完毕
        if all(finished):
            break

    for process in process_list:
        process.join()

    result_dict = read_pkl_files(file_dic_path)

    list_of_tuples = list(result_dict.items())

    with open(os.path.join(filepathstore,list_of_tuples_name),'wb') as file:
        pickle.dump(list_of_tuples,file)
    print("pickle sucesss")




















