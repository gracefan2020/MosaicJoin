"""
LLM-based Entity Linking for Semantic Join

This script performs entity linking by using an LLM to identify matching value pairs
from the contributing entities saved during semantic join retrieval.

It loads the contributing_entities.csv files (which contain value pairs from sketches
that led to column matches during retrieval) and asks the LLM to identify which
value pairs actually represent the same entity.

Output: CSV files with schema:
  query_table, query_column, query_value, candidate_table, candidate_column, candidate_value
"""

import pandas as pd
from typing import Dict, List, Tuple, Set
import glob
import json
import os
import time
import argparse
import random
from pathlib import Path
from openai import OpenAI

MODEL = "openai/gpt-oss-20b"
DEEPINFRA_API_KEY = "VtgGIBPqTAZKNsQrzerTy2YuzHeh4bXk"


def load_contributing_entities_for_query(query_results_dir: str, query_table: str, 
                                          query_column: str, candidate_table: str,
                                          candidate_column: str) -> pd.DataFrame:
    """Load contributing entities for a specific query-candidate pair.
    
    Args:
        query_results_dir: Directory containing query results (with job_* subdirs)
        query_table: Query table name
        query_column: Query column name
        candidate_table: Candidate table name
        candidate_column: Candidate column name
        
    Returns:
        DataFrame with contributing entities, or empty DataFrame if not found
    """
    # Contributing entities files follow the pattern:
    # query_{i:03d}_{query_table}_{query_column}_{candidate_table}_{candidate_column}_contributing_entities.csv
    
    # Search in all job directories
    pattern = f"**/query_*_{query_table}_{query_column}_{candidate_table}_{candidate_column}_contributing_entities.csv"
    matching_files = list(Path(query_results_dir).glob(pattern))
    
    if not matching_files:
        # Try with .csv extension variations
        pattern2 = f"**/query_*_{query_table}.csv_{query_column}_{candidate_table}_{candidate_column}_contributing_entities.csv"
        matching_files = list(Path(query_results_dir).glob(pattern2))
    
    if not matching_files:
        return pd.DataFrame()
    
    # Load the first matching file
    return pd.read_csv(matching_files[0])


def load_all_contributing_entities(query_results_dir: str) -> Dict[Tuple[str, str, str, str], pd.DataFrame]:
    """Load all contributing entities files from query results directory.
    
    Args:
        query_results_dir: Directory containing query results
        
    Returns:
        Dictionary mapping (query_table, query_column, candidate_table, candidate_column) -> DataFrame
    """
    entities_dict = {}
    
    # Find all contributing_entities.csv files
    pattern = "**/contributing_entities/query_*_contributing_entities.csv"
    print(f"Loading contributing entities from {pattern}")
    for filepath in Path(query_results_dir).glob(pattern):
        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                continue
                
            # Extract key from the first row
            row = df.iloc[0]
            query_table = str(row['query_table']).replace('.csv', '').lower()
            query_column = str(row['query_column'])
            candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
            candidate_column = str(row['candidate_column'])
            
            key = (query_table, query_column, candidate_table, candidate_column)
            entities_dict[key] = df
            
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
            continue
    
    print(f"Loaded contributing entities for {len(entities_dict)} column pairs")
    return entities_dict


def build_entity_linking_prompt(query_table: str, query_column: str,
                                 candidate_table: str, candidate_column: str,
                                 value_pairs: List[Tuple[str, str, float]],
                                 max_pairs: int = 50) -> str:
    """Build a prompt for the LLM to identify matching value pairs.
    
    Args:
        query_table: Query table name
        query_column: Query column name
        candidate_table: Candidate table name
        candidate_column: Candidate column name
        value_pairs: List of (query_value, candidate_value, similarity_score) tuples
        max_pairs: Maximum number of pairs to include in prompt
        
    Returns:
        LLM prompt string
    """
    # Limit pairs and sort by similarity (highest first)
    sorted_pairs = sorted(value_pairs, key=lambda x: x[2], reverse=True)[:max_pairs]
    
    # Format pairs for the prompt
    pairs_text = "\n".join([
        f"  {i+1}. \"{qv}\" <-> \"{cv}\" (similarity: {sim:.3f})"
        for i, (qv, cv, sim) in enumerate(sorted_pairs)
    ])
    
    prompt = f"""TASK: Entity Linking - Identify which value pairs refer to the SAME real-world entity.

You are given pairs of values from two database columns. Your task is to identify which pairs represent the SAME entity, even if they have different surface forms.

GUIDELINES:
- Values match if they refer to the SAME real-world entity (person, place, organization, etc.)
- Different syntactic forms are OK: "NYC" matches "New York City", "USA" matches "United States"
- Abbreviations match full names: "MIT" matches "Massachusetts Institute of Technology"
- Partial matches are NOT valid: "New York" does NOT match "New York Times"
- Different entities with similar names do NOT match: "Cambridge, MA" does NOT match "Cambridge, UK"

EXAMPLES OF MATCHING PAIRS:
- "New York City" <-> "NYC" ✓ (same city, different forms)
- "United States of America" <-> "USA" ✓ (same country)
- "Barack Obama" <-> "Obama, Barack" ✓ (same person)
- "Microsoft Corporation" <-> "Microsoft" ✓ (same company)

EXAMPLES OF NON-MATCHING PAIRS:
- "Apple" (company) <-> "Apple" (fruit) ✗ (different entities)
- "Georgia" (US state) <-> "Georgia" (country) ✗ (different places)
- "Paris" <-> "Paris Hilton" ✗ (city vs person)
- "United Poland" <-> "Poland" ✗ (political party vs country - different entity types)

---

Query column: {query_table}.{query_column}
Candidate column: {candidate_table}.{candidate_column}

Value pairs (with embedding similarity scores):
{pairs_text}

Return ONLY the numbers of the pairs that represent the SAME entity, as a comma-separated list.
If no pairs match, return "NONE".

Example response: 1, 3, 5, 7
Or if none match: NONE"""
    print(prompt)
    return prompt


def call_llm(prompt: str) -> str:
    """Call the LLM API with the given prompt."""
    client = OpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url="https://api.deepinfra.com/v1/openai",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Lower temperature for more consistent entity linking
    )
    return response.choices[0].message.content


def parse_llm_response(response: str, value_pairs: List[Tuple[str, str, float]]) -> List[Tuple[str, str]]:
    """Parse LLM response to extract matching value pairs.
    
    Args:
        response: LLM response string
        value_pairs: Original list of (query_value, candidate_value, similarity) tuples
        
    Returns:
        List of (query_value, candidate_value) tuples that the LLM identified as matches
    """
    response = response.strip().upper()
    
    if "NONE" in response:
        return []
    
    # Extract numbers from the response
    import re
    numbers = re.findall(r'\d+', response)
    
    matched_pairs = []
    for num_str in numbers:
        idx = int(num_str) - 1  # Convert to 0-based index
        if 0 <= idx < len(value_pairs):
            qv, cv, _ = value_pairs[idx]
            matched_pairs.append((qv, cv))
    
    return matched_pairs


def load_llm_checkpoint(checkpoint_file: str) -> Dict[Tuple[str, str, str, str], List[Tuple[str, str]]]:
    """Load existing LLM entity linking results from checkpoint file."""
    if not os.path.exists(checkpoint_file):
        return {}
    
    df = pd.read_csv(checkpoint_file)
    checkpoint = {}
    
    for _, row in df.iterrows():
        query_table = str(row['query_table']).replace('.csv', '').lower()
        candidate_table = str(row['candidate_table']).replace('.csv', '').lower()
        key = (query_table, row['query_column'], candidate_table, row['candidate_column'])
        
        if key not in checkpoint:
            checkpoint[key] = []
        
        checkpoint[key].append((row['query_value'], row['candidate_value']))
    
    print(f"Loaded checkpoint with {len(checkpoint)} column pairs processed")
    return checkpoint


def save_entity_link(output_file: str, query_table: str, query_column: str,
                     candidate_table: str, candidate_column: str,
                     query_value: str, candidate_value: str):
    """Append a single entity link to the output file."""
    new_row = {
        'query_table': query_table,
        'query_column': query_column,
        'query_value': query_value,
        'candidate_table': candidate_table,
        'candidate_column': candidate_column,
        'candidate_value': candidate_value
    }
    
    file_exists = os.path.exists(output_file)
    df = pd.DataFrame([new_row])
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)


def main():
    parser = argparse.ArgumentParser(description='LLM-based entity linking for semantic join results')
    parser.add_argument('query_results_dir', type=str,
                       help='Directory containing the query results (with contributing_entities.csv files)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output CSV file for entity links (default: {query_results_dir}/llm_entity_links.csv)')
    parser.add_argument('--max-pairs-per-prompt', type=int, default=50,
                       help='Maximum number of value pairs to include in each LLM prompt')
    parser.add_argument('--query-indices', type=int, nargs='*',
                       help='Specific query indices to process (0-based)')
    args = parser.parse_args()
    
    query_results_dir = args.query_results_dir
    output_file = args.output_file or f"{query_results_dir}/llm_entity_links.csv"
    
    # Load all contributing entities
    all_entities = load_all_contributing_entities(query_results_dir)
    
    if not all_entities:
        print("No contributing entities found. Make sure the query results directory contains contributing_entities.csv files.")
        return
    
    # Load checkpoint to resume from
    checkpoint = load_llm_checkpoint(output_file)
    
    # Track statistics
    total_pairs_processed = 0
    total_matches_found = 0
    llm_call_times = []
    
    # Process each column pair
    column_pairs = list(all_entities.keys())
    
    # Filter by query indices if specified
    if args.query_indices:
        filtered_pairs = [column_pairs[i] for i in args.query_indices if i < len(column_pairs)]
        column_pairs = filtered_pairs
        print(f"Processing {len(column_pairs)} specified column pairs")
    
    for key in column_pairs:
        query_table, query_column, candidate_table, candidate_column = key
        
        # Skip if already processed
        if key in checkpoint:
            print(f"Skipping {query_table}.{query_column} -> {candidate_table}.{candidate_column} (already processed)")
            continue
        
        print(f"\nProcessing: {query_table}.{query_column} -> {candidate_table}.{candidate_column}")
        
        # Get contributing entities for this pair
        entities_df = all_entities[key]
        
        # Extract value pairs with similarity scores
        value_pairs = [
            (str(row['query_value']), str(row['candidate_value']), float(row['similarity_score']))
            for _, row in entities_df.iterrows()
        ]
        
        if not value_pairs:
            print("  No value pairs found, skipping")
            continue
        
        print(f"  Found {len(value_pairs)} value pairs from sketches")
        
        # Build prompt and call LLM
        prompt = build_entity_linking_prompt(
            query_table, query_column,
            candidate_table, candidate_column,
            value_pairs,
            max_pairs=args.max_pairs_per_prompt
        )
        
        start_time = time.time()
        try:
            llm_response = call_llm(prompt)
            call_time = time.time() - start_time
            llm_call_times.append(call_time)
            
            print(f"  LLM response (took {call_time:.2f}s): {llm_response[:100]}...")
            
            # Parse response to get matching pairs
            sorted_pairs = sorted(value_pairs, key=lambda x: x[2], reverse=True)[:args.max_pairs_per_prompt]
            matched_pairs = parse_llm_response(llm_response, sorted_pairs)
            
            print(f"  Found {len(matched_pairs)} entity links")
            
            # Save matched pairs
            for query_value, candidate_value in matched_pairs:
                save_entity_link(
                    output_file,
                    query_table, query_column,
                    candidate_table, candidate_column,
                    query_value, candidate_value
                )
            
            total_pairs_processed += len(value_pairs)
            total_matches_found += len(matched_pairs)
            
        except Exception as e:
            print(f"  Error calling LLM: {e}")
            continue
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"Entity Linking Summary:")
    print(f"  Column pairs processed: {len(column_pairs)}")
    print(f"  Total value pairs processed: {total_pairs_processed}")
    print(f"  Total entity links found: {total_matches_found}")
    
    if llm_call_times:
        print(f"\nLLM Call Statistics:")
        print(f"  Total LLM calls: {len(llm_call_times)}")
        print(f"  Total time: {sum(llm_call_times):.2f}s")
        print(f"  Average time per call: {sum(llm_call_times)/len(llm_call_times):.2f}s")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
