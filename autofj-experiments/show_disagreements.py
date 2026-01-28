#!/usr/bin/env python3
"""Simple script to show disagreements between SemSketch and DeepJoin+AutoFJ."""

import csv
import glob
from pathlib import Path
from collections import defaultdict

def normalize(s):
    """Normalize string for comparison."""
    return str(s).lower().strip().replace('.csv', '')

def load_matches_from_csv(csv_path):
    """Load matches from a single CSV file."""
    matches = set()
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_table = normalize(row.get('query_table', ''))
                q_val = normalize(row.get('query_value', ''))
                c_table = normalize(row.get('candidate_table', ''))
                c_val = normalize(row.get('candidate_value', ''))
                if q_table and q_val and c_table and c_val:
                    matches.add((q_table, q_val, c_table, c_val))
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
    return matches

def load_matches_from_dir(dir_path):
    """Load all matches from match files in a directory.
    
    First tries to find _matches.csv files. If none found, falls back to
    _contributing_entities.csv files (for SemSketch), which have the same format.
    """
    matches = set()
    dir_path = Path(dir_path)
    # Find all match files recursively
    match_files = list(dir_path.glob("**/query_*_matches.csv"))
    
    if match_files:
        print(f"Found {len(match_files)} match files in {dir_path}")
        for match_file in match_files:
            matches.update(load_matches_from_csv(match_file))
    else:
        # Fall back to contributing entities files (SemSketch format)
        # These files have the same CSV format as match files!
        contributing_files = list(dir_path.glob("**/*_contributing_entities.csv"))
        print(f"No match files found. Found {len(contributing_files)} contributing entities files in {dir_path}")
        print("Note: Using contributing entities files (evidence pairs that supported retrieval).")
        print("      Match files may not have been generated if no matches passed the similarity threshold.")
        
        for contrib_file in contributing_files:
            matches.update(load_matches_from_csv(contrib_file))
    
    return matches

def load_ground_truth(csv_path):
    """Load ground truth from CSV.
    
    Format: id_l, title_l, id_r, title_r, dataset
    Need to map dataset to table names using groundtruth-joinable.csv
    """
    matches = set()
    try:
        # Load joinability GT to map dataset to table names
        joinable_path = Path(csv_path).parent / "groundtruth-joinable.csv"
        dataset_to_tables = {}
        if joinable_path.exists():
            with open(joinable_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = normalize(row.get('dataset', ''))
                    left_table = normalize(row.get('left_table', ''))
                    right_table = normalize(row.get('right_table', ''))
                    if dataset:
                        dataset_to_tables[dataset] = (left_table, right_table)
        
        # Load entity linking GT
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = normalize(row.get('dataset', ''))
                q_val = normalize(row.get('title_l', ''))
                c_val = normalize(row.get('title_r', ''))
                
                if dataset in dataset_to_tables:
                    left_table, right_table = dataset_to_tables[dataset]
                    # Add both directions: left->right and right->left
                    matches.add((left_table, q_val, right_table, c_val))
                    matches.add((right_table, c_val, left_table, q_val))
    except Exception as e:
        print(f"Error loading GT {csv_path}: {e}")
        import traceback
        traceback.print_exc()
    return matches

# Paths
semsketch_dir = "autofj_query_results_k1024_t0.1_top10_slurm"
deepjoin_dir = "autofj_deepjoin_baseline_k10_n10_t0.1"
gt_path = "../datasets/autofj_join_benchmark/groundtruth-entity-linking.csv"

print("Loading files...")
print("Loading SemSketch matches...")
semsketch_matches = load_matches_from_dir(semsketch_dir)
print("Loading DeepJoin+AutoFJ matches...")
deepjoin_matches = load_matches_from_dir(deepjoin_dir)
print("Loading ground truth...")
gt_matches = load_ground_truth(gt_path)

print(f"SemSketch matches: {len(semsketch_matches)}")
print(f"DeepJoin matches: {len(deepjoin_matches)}")
print(f"Ground truth matches: {len(gt_matches)}")

# Build index of matches by query value for each method
# Format: {(query_table, query_value): set of (candidate_table, candidate_value)}
semsketch_by_query = defaultdict(set)
for q_table, q_val, c_table, c_val in semsketch_matches:
    semsketch_by_query[(q_table, q_val)].add((c_table, c_val))

deepjoin_by_query = defaultdict(set)
for q_table, q_val, c_table, c_val in deepjoin_matches:
    deepjoin_by_query[(q_table, q_val)].add((c_table, c_val))

# Build reverse index: by candidate value (to find what query values match a given candidate value)
# Format: {(candidate_table, candidate_value): set of (query_table, query_value)}
semsketch_by_candidate = defaultdict(set)
for q_table, q_val, c_table, c_val in semsketch_matches:
    semsketch_by_candidate[(c_table, c_val)].add((q_table, q_val))

deepjoin_by_candidate = defaultdict(set)
for q_table, q_val, c_table, c_val in deepjoin_matches:
    deepjoin_by_candidate[(c_table, c_val)].add((q_table, q_val))

# Find correct matches for each method
semsketch_correct = semsketch_matches & gt_matches
deepjoin_correct = deepjoin_matches & gt_matches

print(f"\nSemSketch correct: {len(semsketch_correct)}")
print(f"DeepJoin correct: {len(deepjoin_correct)}")

# Find disagreements
deepjoin_correct_semsketch_wrong = deepjoin_correct - semsketch_correct
semsketch_correct_deepjoin_wrong = semsketch_correct - deepjoin_correct

print(f"\n{'='*80}")
print("DISAGREEMENTS")
print(f"{'='*80}")
print(f"\nDeepJoin+AutoFJ correct, SemSketch wrong: {len(deepjoin_correct_semsketch_wrong)}")
print(f"SemSketch correct, DeepJoin+AutoFJ wrong: {len(semsketch_correct_deepjoin_wrong)}")

print(f"\n{'='*80}")
print("EXAMPLES: DeepJoin+AutoFJ correct, SemSketch wrong")
print(f"{'='*80}")
examples = list(deepjoin_correct_semsketch_wrong)[:10]
for i, (q_table, q_val, c_table, c_val) in enumerate(examples):
    print(f"\n{i+1}. Query: {q_table}['{q_val}']")
    print(f"   Ground truth match: {c_table}['{c_val}']")
    print(f"   ✓ DeepJoin+AutoFJ found this match")
    print(f"   ✗ SemSketch missed this match")
    
    # Show what SemSketch actually found for this query value
    semsketch_alternatives = semsketch_by_query.get((q_table, q_val), set())
    if semsketch_alternatives:
        print(f"   SemSketch found {len(semsketch_alternatives)} alternative match(es) for this query value:")
        for alt_c_table, alt_c_val in list(semsketch_alternatives)[:3]:  # Show up to 3 alternatives
            is_correct = (q_table, q_val, alt_c_table, alt_c_val) in gt_matches
            marker = "✓" if is_correct else "✗"
            print(f"     {marker} {alt_c_table}['{alt_c_val}']")
        if len(semsketch_alternatives) > 3:
            print(f"     ... and {len(semsketch_alternatives) - 3} more")
    else:
        print(f"   SemSketch found no matches for this query value")
        # Check if the candidate value appears as a query value in SemSketch's results
        semsketch_reverse_matches = semsketch_by_candidate.get((c_table, c_val), set())
        if semsketch_reverse_matches:
            print(f"   However, SemSketch found {len(semsketch_reverse_matches)} match(es) where '{c_val}' is the query value:")
            for alt_q_table, alt_q_val in list(semsketch_reverse_matches)[:3]:  # Show up to 3
                is_correct = (alt_q_table, alt_q_val, c_table, c_val) in gt_matches
                marker = "✓" if is_correct else "✗"
                print(f"     {marker} {alt_q_table}['{alt_q_val}'] -> {c_table}['{c_val}']")
            if len(semsketch_reverse_matches) > 3:
                print(f"     ... and {len(semsketch_reverse_matches) - 3} more")

if len(deepjoin_correct_semsketch_wrong) > 10:
    print(f"\n... and {len(deepjoin_correct_semsketch_wrong) - 10} more examples")

print(f"\n{'='*80}")
print("EXAMPLES: SemSketch correct, DeepJoin+AutoFJ wrong")
print(f"{'='*80}")
examples = list(semsketch_correct_deepjoin_wrong)[:10]
for i, (q_table, q_val, c_table, c_val) in enumerate(examples):
    print(f"\n{i+1}. Query: {q_table}['{q_val}']")
    print(f"   Ground truth match: {c_table}['{c_val}']")
    print(f"   ✓ SemSketch found this match")
    print(f"   ✗ DeepJoin+AutoFJ missed this match")
    
    # Show what DeepJoin+AutoFJ actually found for this query value
    deepjoin_alternatives = deepjoin_by_query.get((q_table, q_val), set())
    if deepjoin_alternatives:
        print(f"   DeepJoin+AutoFJ found {len(deepjoin_alternatives)} alternative match(es) for this query value:")
        for alt_c_table, alt_c_val in list(deepjoin_alternatives)[:3]:  # Show up to 3 alternatives
            is_correct = (q_table, q_val, alt_c_table, alt_c_val) in gt_matches
            marker = "✓" if is_correct else "✗"
            print(f"     {marker} {alt_c_table}['{alt_c_val}']")
        if len(deepjoin_alternatives) > 3:
            print(f"     ... and {len(deepjoin_alternatives) - 3} more")
    else:
        print(f"   DeepJoin+AutoFJ found no matches for this query value")
        # Check if the candidate value appears as a query value in DeepJoin's results
        deepjoin_reverse_matches = deepjoin_by_candidate.get((c_table, c_val), set())
        if deepjoin_reverse_matches:
            print(f"   However, DeepJoin+AutoFJ found {len(deepjoin_reverse_matches)} match(es) where '{c_val}' is the query value:")
            for alt_q_table, alt_q_val in list(deepjoin_reverse_matches)[:3]:  # Show up to 3
                is_correct = (alt_q_table, alt_q_val, c_table, c_val) in gt_matches
                marker = "✓" if is_correct else "✗"
                print(f"     {marker} {alt_q_table}['{alt_q_val}'] -> {c_table}['{c_val}']")
            if len(deepjoin_reverse_matches) > 3:
                print(f"     ... and {len(deepjoin_reverse_matches) - 3} more")

if len(semsketch_correct_deepjoin_wrong) > 10:
    print(f"\n... and {len(semsketch_correct_deepjoin_wrong) - 10} more examples")

