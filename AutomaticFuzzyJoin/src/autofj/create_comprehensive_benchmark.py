import pandas as pd
import os
from os.path import dirname

def create_comprehensive_benchmark():
    benchmark_dir = os.path.join(dirname(__file__), "benchmark")
    join_benchmark_dir = os.path.join(dirname(__file__), "join_benchmark")
    os.makedirs(join_benchmark_dir, exist_ok=True)
    
    all_gt = []
    joinable_tables = []
    
    for dataset_name in os.listdir(benchmark_dir):
        dataset_path = os.path.join(benchmark_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
            
        # Copy and rename left and right tables
        left_path = os.path.join(dataset_path, "left.csv")
        right_path = os.path.join(dataset_path, "right.csv")
        gt_path = os.path.join(dataset_path, "gt.csv")
        
        if os.path.exists(left_path):
            pd.read_csv(left_path).to_csv(
                os.path.join(join_benchmark_dir, f"{dataset_name}_left.csv"), 
                index=False
            )
        if os.path.exists(right_path):
            pd.read_csv(right_path).to_csv(
                os.path.join(join_benchmark_dir, f"{dataset_name}_right.csv"), 
                index=False
            )
        
        # Collect groundtruth with dataset column
        if os.path.exists(gt_path):
            gt = pd.read_csv(gt_path)
            gt["dataset"] = dataset_name
            all_gt.append(gt)
            joinable_tables.append({
                "dataset": dataset_name,
                "left_table": f"{dataset_name}_left.csv",
                "right_table": f"{dataset_name}_right.csv"
            })
    
    # Create combined groundtruth table
    combined_gt = pd.concat(all_gt, ignore_index=True)
    combined_gt.to_csv(
        os.path.join(join_benchmark_dir, "groundtruth.csv"), 
        index=False
    )
    
    # Create joinable tables metadata
    joinable_df = pd.DataFrame(joinable_tables)
    joinable_df.to_csv(
        os.path.join(join_benchmark_dir, "joinable_tables.csv"), 
        index=False
    )


if __name__ == "__main__":
    create_comprehensive_benchmark()