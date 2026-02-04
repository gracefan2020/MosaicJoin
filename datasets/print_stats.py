#!/usr/bin/env python3
"""Print dataset statistics for the merged AutoFJ-GDC benchmark."""

import os
import pandas as pd
from pathlib import Path

def get_csv_stats(csv_path: str) -> dict:
    """Get number of columns and rows for a CSV file."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        return {"columns": len(df.columns), "rows": len(df)}
    except Exception as e:
        print(f"Warning: Could not read {csv_path}: {e}")
        return {"columns": 0, "rows": 0}

def main():
    base_dir = "santos-small"
    datalake_dir = Path(base_dir) / "datalake_singletons"
    print(datalake_dir)
    # Get query table statistics from gdc_breakdown_query_columns.csv
    # breakdown_csv = Path(base_dir) / "wt_query_columns_no_column_names.csv"
    breakdown_csv = "wt-autofj/wt_query_columns_no_column_names.csv"

    breakdown_df = pd.read_csv(breakdown_csv)
    num_query_tables = breakdown_df["target_ds"].nunique()

    # For each unique target_ds, count the number of rows (attributes) corresponding to that ds
    query_columns = breakdown_df.groupby("target_ds")["target_attr"].count().tolist()
    avg_query_columns = sum(query_columns) / len(query_columns) if query_columns else 0

    # Get datalake table statistics
    datalake_files = list(datalake_dir.glob("*.csv"))
    num_datalake_tables = len(datalake_files)
    
    total_columns = 0
    total_rows = 0
    all_columns = []
    all_rows = []
    
    for df_path in datalake_files:
        stats = get_csv_stats(df_path)
        total_columns += stats["columns"]
        total_rows += stats["rows"]
        all_columns.append(stats["columns"])
        all_rows.append(stats["rows"])
    
    avg_columns = total_columns / num_datalake_tables if num_datalake_tables > 0 else 0
    avg_rows = total_rows / num_datalake_tables if num_datalake_tables > 0 else 0
    
    # Print statistics
    print("=" * 60)
    print(base_dir + " Benchmark Statistics")
    print("=" * 60)
    # print(f"# Query Tables:        {num_query_tables}")
    print(f"Total # Query Columns:   {sum(query_columns)}")
    print("-" * 60)
    print(f"# Datalake Tables:     {num_datalake_tables}")
    print(f"Total # Rows:          {total_rows}")
    print(f"Average # Rows:        {avg_rows:.2f}")
    print("=" * 60)
    
    # Print in table format for easy copy-paste
    print("\nTable format (for paper):")
    print("-" * 60)
    print(f"| Total # Query Columns | # Datalake Tables | Total # Rows | Avg # Rows |")
    print(f"| {sum(query_columns)} | {num_datalake_tables} | {total_rows} | {avg_rows:.2f} |")

if __name__ == "__main__":
    main()
