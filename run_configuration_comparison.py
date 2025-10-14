#!/usr/bin/env python3
"""
Example: Running Different Configurations

This script demonstrates how to run the semantic join pipeline with different
configurations and compare results.
"""

import os
import subprocess
from pathlib import Path

def run_configuration(sketch_size, threshold, top_k, description):
    """Run the pipeline with a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Running Configuration: {description}")
    print(f"Sketch Size: {sketch_size}, Threshold: {threshold}, Top-K: {top_k}")
    print(f"{'='*60}")
    
    # Update run_queries.py with new parameters
    with open("run_queries.py", "r") as f:
        content = f.read()
    
    # Replace the configuration parameters
    content = content.replace("sketch_size = 1024", f"sketch_size = {sketch_size}")
    content = content.replace("similarity_threshold = 0.7", f"similarity_threshold = {threshold}")
    content = content.replace("top_k_return = 50", f"top_k_return = {top_k}")
    
    with open("run_queries.py", "w") as f:
        f.write(content)
    
    # Run query processing
    print("Running query processing...")
    result = subprocess.run(["python", "run_queries.py"], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Query processing completed successfully")
        
        # Run evaluation
        print("Running evaluation...")
        eval_result = subprocess.run(["python", "run_evaluation.py"], capture_output=True, text=True)
        
        if eval_result.returncode == 0:
            print("✅ Evaluation completed successfully")
        else:
            print(f"❌ Evaluation failed: {eval_result.stderr}")
    else:
        print(f"❌ Query processing failed: {result.stderr}")

def main():
    """Run different configurations and compare results."""
    
    # Check if required files exist
    if not Path("offline_data/embeddings").exists():
        print("Error: Embeddings not found. Please run:")
        print("  python run_offline_embeddings_parallel.py")
        return 1
    
    if not Path("offline_data/sketches_k1024").exists():
        print("Error: Sketches not found. Please run:")
        print("  python run_offline_sketch_parallel.py")
        return 1
    
    # Define configurations to test
    configurations = [
        (1024, 0.7, 50, "Default Configuration"),
        (512, 0.7, 50, "Smaller Sketch Size"),
        (2048, 0.7, 50, "Larger Sketch Size"),
        (1024, 0.5, 50, "Lower Threshold"),
        (1024, 0.8, 50, "Higher Threshold"),
        (1024, 0.7, 20, "Fewer Results"),
        (1024, 0.7, 100, "More Results"),
    ]
    
    print("🚀 Starting Configuration Comparison")
    print("This will run multiple configurations and save results separately")
    
    for sketch_size, threshold, top_k, description in configurations:
        run_configuration(sketch_size, threshold, top_k, description)
    
    print(f"\n{'='*60}")
    print("🎉 All configurations completed!")
    print("Check the following directories for results:")
    
    # List all result directories
    query_dirs = [d for d in os.listdir(".") if d.startswith("query_results_k")]
    eval_dirs = [d for d in os.listdir(".") if d.startswith("evaluation_results_k")]
    
    print("\nQuery Results:")
    for dir_name in sorted(query_dirs):
        print(f"  📁 {dir_name}")
    
    print("\nEvaluation Results:")
    for dir_name in sorted(eval_dirs):
        print(f"  📁 {dir_name}")
    
    print(f"\n{'='*60}")
    print("To compare results, check the evaluation_summary.json files in each directory")
    
    return 0

if __name__ == "__main__":
    exit(main())
