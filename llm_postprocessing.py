import pandas as pd
from evaluate_semantic_join import load_semantic_results
from typing import Dict, List, Tuple, Set
import glob
import json
import os
import time
import argparse
from openai import OpenAI

MODEL = "openai/gpt-oss-20b"
DEEPINFRA_API_KEY = "VtgGIBPqTAZKNsQrzerTy2YuzHeh4bXk"

datalake_dir = "datasets/freyja-semantic-join/datalake"

def load_all_datalake_tables() -> List[str]:
    table_dfs = {}
    all_datalake_tables = glob.glob(f"{datalake_dir}/*.csv")
    for table in all_datalake_tables:
        df = pd.read_csv(table, low_memory=False)
        table_dfs[table.split("/")[-1].replace(".csv", "").lower()] = df
    return table_dfs

def load_semantic_preds(query_results_dir: str) -> Dict[str, List[str]]:
    _, semantic_matches = load_semantic_results(f"{query_results_dir}/all_query_results.csv", 0.7)
    sketch_results = {q: [c for c in semantic_matches[q].keys()] for q in semantic_matches.keys()}
    return sketch_results

def load_original_results(query_results_dir: str):
    """Load the original results CSV to extract full metadata."""
    return pd.read_csv(f"{query_results_dir}/all_query_results.csv")

def get_table_info(table_name: str, column_name: str, all_datalake_table_dfs: Dict[str, pd.DataFrame]) -> Tuple[List[str], str, List[str]]:
    # return the schema, the first 5 rows, and a sample of values for a given column
    # format the first 5 rows as a string in markdown table format
    df = all_datalake_table_dfs[table_name]
    table_rows_str = "| " + " | ".join(df.columns.tolist()) + " |\n| " + " | ".join(["---"] * len(df.columns)) + " |\n"
    for row in df.head(5).values.tolist():
        table_rows_str += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    return df.columns.tolist(), table_rows_str, df[column_name].unique().tolist()

def build_llm_prompt(query_table_name: str, query_column_name: str, query_schema: List[str], query_table_rows: str, query_col_vals: List[str], candidate_table_name: str, candidate_column_name: str, candidate_schema: List[str], candidate_table_rows: str, candidate_col_vals: List[str], max_values: int = 100) -> str:
   
    prompt = f"""TASK: Determine if these columns are semantically joinable.
Two columns are JOINABLE if:
1. Same CONCEPT (e.g., cities, names, IDs)
2. Substantial VALUE OVERLAP (same entities, different forms: "NYC"="New York City"="Big Apple")
3. Tables may differ (airports vs hospitals joinable on city columns)

Query: table="{query_table_name}", column="{query_column_name}", schema={query_schema}
Values: {query_col_vals[:max_values]}

Candidate: table="{candidate_table_name}", column="{candidate_column_name}", schema={candidate_schema}
Values: {candidate_col_vals[:max_values]}

Return "Yes" if semantically joinable, "No" otherwise."""
    return prompt

def call_llm(prompt: str) -> str:
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )   
    return response.choices[0].message.content

def is_approved(response: str) -> bool:
    """Check if LLM response indicates approval (Yes/True/etc)."""
    response_lower = response.strip().lower()
    return "yes" in response_lower or "true" in response_lower or response_lower == "y"

def load_llm_checkpoint(checkpoint_file: str) -> Dict[Tuple[str, str], Tuple[str, bool]]:
    """Load existing LLM responses from checkpoint file."""
    if not os.path.exists(checkpoint_file):
        return {}
    
    df = pd.read_csv(checkpoint_file)
    checkpoint = {}
    for _, row in df.iterrows():
        query_key = f"{row['query_table']}.{row['query_column']}"
        candidate_key = f"{row['candidate_table']}.{row['candidate_column']}"
        llm_response = row['llm_response']
        approved = bool(row['approved'])
        checkpoint[(query_key, candidate_key)] = (llm_response, approved)
    
    print(f"Loaded {len(checkpoint)} existing LLM responses from checkpoint")
    return checkpoint

def save_llm_checkpoint(checkpoint_file: str, query_table: str, query_column: str, 
                        candidate_table: str, candidate_column: str, 
                        llm_response: str, approved: bool):
    """Append a single LLM response to the checkpoint file."""
    new_row = {
        'query_table': query_table,
        'query_column': query_column,
        'candidate_table': candidate_table,
        'candidate_column': candidate_column,
        'llm_response': llm_response,
        'approved': approved
    }
    
    # Check if file exists to determine if we need headers
    file_exists = os.path.exists(checkpoint_file)
    
    df = pd.DataFrame([new_row])
    df.to_csv(checkpoint_file, mode='a', header=not file_exists, index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='LLM post-processing for semantic join results')
    parser.add_argument('query_results_dir', type=str, 
                       help='Directory containing the query results CSV')
    args = parser.parse_args()
    
    query_results_dir = args.query_results_dir
    
    all_datalake_table_dfs = load_all_datalake_tables()
    sketch_results = load_semantic_preds(query_results_dir)
    original_results_df = load_original_results(query_results_dir)

    # LLM checkpoint files
    checkpoint_file = f"{query_results_dir}/llm_responses_checkpoint.csv"
    saved_results_csv = f"{query_results_dir}/llm_pruned_query_results.csv"
    
    # Load existing checkpoint
    llm_checkpoint = load_llm_checkpoint(checkpoint_file)
    llm_approved_pairs = set()  # Store (query, candidate) pairs
    
    # Track LLM call timing
    llm_call_times = []
    total_llm_calls = 0
    
    for q, candidates in sketch_results.items():
        query_table = q.split(".")[0]
        query_column = q.split(".")[1]
        query_schema, query_table_rows, query_col_vals = get_table_info(query_table, query_column, all_datalake_table_dfs)
        
        for c in candidates:
            candidate_table = c.split(".")[0]
            candidate_column = c.split(".")[1]
            
            # Check if we already have this response in checkpoint
            if (q, c) in llm_checkpoint:
                llm_response, approved = llm_checkpoint[(q, c)]
                print(f"Using cached response for {q} -> {c}: {llm_response}")
                if approved:
                    llm_approved_pairs.add((q, c))
            else:
                # Need to call LLM
                candidate_schema, candidate_table_rows, candidate_col_vals = get_table_info(candidate_table, candidate_column, all_datalake_table_dfs)
                llm_prompt = build_llm_prompt(query_table, query_column, query_schema, query_table_rows, query_col_vals, candidate_table, candidate_column, candidate_schema, candidate_table_rows, candidate_col_vals)
                print(f"Query Table: {query_table}, Query Column: {query_column}, Candidate Table: {candidate_table}, Candidate Column: {candidate_column}")
                print("-"*100)
                
                # Time the LLM call
                start_time = time.time()
                llm_response = call_llm(llm_prompt)
                call_time = time.time() - start_time
                llm_call_times.append(call_time)
                total_llm_calls += 1
                
                print(f"LLM Response: {llm_response} (took {call_time:.2f}s)")
                
                approved = is_approved(llm_response)
                
                # Save to checkpoint immediately
                save_llm_checkpoint(checkpoint_file, query_table, query_column, 
                                   candidate_table, candidate_column, 
                                   llm_response, approved)
                
                # Update checkpoint dict
                llm_checkpoint[(q, c)] = (llm_response, approved)
                
                if approved:
                    llm_approved_pairs.add((q, c))
                
                print("-"*100)
                
    # Filter original results to only include LLM-approved pairs
    pruned_rows = []
    for _, row in original_results_df.iterrows():
        query_table = str(row['query_table']).replace('.csv', '').lower()
        candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
        query_key = f"{query_table}.{row['query_column']}"
        candidate_key = f"{candidate_table}.{row['candidate_column']}"
        
        if (query_key, candidate_key) in llm_approved_pairs:
            pruned_rows.append(row)
    
    # Save pruned results
    pruned_df = pd.DataFrame(pruned_rows)
    pruned_df.to_csv(saved_results_csv, index=False)
    print(f"Saved {len(pruned_df)} pruned results to {saved_results_csv}")
    
    # Print timing statistics
    if llm_call_times:
        avg_time = sum(llm_call_times) / len(llm_call_times)
        total_time = sum(llm_call_times)
        print(f"\n{'='*80}")
        print(f"LLM Call Statistics:")
        print(f"  Total LLM calls made: {total_llm_calls}")
        print(f"  Total time for LLM calls: {total_time:.2f}s")
        print(f"  Average time per LLM call: {avg_time:.2f}s")
        print(f"  Minimum call time: {min(llm_call_times):.2f}s")
        print(f"  Maximum call time: {max(llm_call_times):.2f}s")
        print(f"{'='*80}")