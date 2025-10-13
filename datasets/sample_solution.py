#!/usr/bin/env python3
"""
Sample solution for the Freyja semantic join analysis notebook
This file contains the implementations for jaccard_similarity and set_containment functions
"""

import pandas as pd
import numpy as np
from typing import Tuple

def jaccard_similarity(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Calculate Jaccard similarity between two pandas Series.
    
    Args:
        series_a: First pandas Series
        series_b: Second pandas Series
    
    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    # Normalize both series: remove special chars, lowercase, strip whitespace
    series_a_norm = series_a.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    series_b_norm = series_b.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    
    # Convert to sets to get unique values (immune to deduplication)
    set_a = set(series_a_norm.dropna())
    set_b = set(series_b_norm.dropna())
    
    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def set_containment(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float]:
    """
    Calculate set containment between two pandas Series.
    
    Args:
        series_a: First pandas Series
        series_b: Second pandas Series
    
    Returns:
        Tuple[float, float]: (forward_containment, backward_containment)
            - forward_containment: |A ∩ B| / |A| (how much of A is in B)
            - backward_containment: |A ∩ B| / |B| (how much of B is in A)
    """
    # Normalize both series (same as Jaccard)
    series_a_norm = series_a.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    series_b_norm = series_b.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    
    # Convert to sets
    set_a = set(series_a_norm.dropna())
    set_b = set(series_b_norm.dropna())
    
    # Calculate intersection
    intersection = set_a.intersection(set_b)
    
    # Calculate forward containment: |intersection| / |set_a|
    forward_containment = len(intersection) / len(set_a) if len(set_a) > 0 else 0.0
    
    # Calculate backward containment: |intersection| / |set_b|
    backward_containment = len(intersection) / len(set_b) if len(set_b) > 0 else 0.0
    
    return forward_containment, backward_containment

def test_implementations():
    """Test the implementations with sample data"""
    print("Testing implementations...")
    
    # Test Jaccard similarity
    print("\n=== Testing Jaccard Similarity ===")
    
    # Test case 1: Identical sets
    series1 = pd.Series(['USA', 'Canada', 'Mexico'])
    series2 = pd.Series(['USA', 'Canada', 'Mexico'])
    result1 = jaccard_similarity(series1, series2)
    print(f"Test 1 - Identical sets: {result1} (expected: 1.0)")
    
    # Test case 2: No overlap
    series3 = pd.Series(['USA', 'Canada'])
    series4 = pd.Series(['France', 'Germany'])
    result2 = jaccard_similarity(series3, series4)
    print(f"Test 2 - No overlap: {result2} (expected: 0.0)")
    
    # Test case 3: Partial overlap
    series5 = pd.Series(['USA', 'Canada', 'Mexico'])
    series6 = pd.Series(['USA', 'France', 'Germany'])
    result3 = jaccard_similarity(series5, series6)
    print(f"Test 3 - Partial overlap: {result3} (expected: ~0.2)")
    
    # Test case 4: With special characters and case differences
    series7 = pd.Series(['USA', 'Canada,', 'Mexico'])
    series8 = pd.Series(['usa', 'Canada', 'France'])
    result4 = jaccard_similarity(series7, series8)
    print(f"Test 4 - Special chars/case: {result4} (expected: ~0.5)")
    
    # Test Set Containment
    print("\n=== Testing Set Containment ===")
    
    # Test case 1: A is subset of B
    series1 = pd.Series(['USA', 'Canada'])
    series2 = pd.Series(['USA', 'Canada', 'Mexico', 'France'])
    forward, backward = set_containment(series1, series2)
    print(f"Test 1 - A subset of B: forward={forward:.2f}, backward={backward:.2f}")
    print(f"  Expected: forward=1.0, backward=0.5")
    
    # Test case 2: No overlap
    series3 = pd.Series(['USA', 'Canada'])
    series4 = pd.Series(['France', 'Germany'])
    forward, backward = set_containment(series3, series4)
    print(f"Test 2 - No overlap: forward={forward:.2f}, backward={backward:.2f}")
    print(f"  Expected: forward=0.0, backward=0.0")
    
    # Test case 3: Empty sets
    series5 = pd.Series([])
    series6 = pd.Series(['USA', 'Canada'])
    forward, backward = set_containment(series5, series6)
    print(f"Test 3 - Empty set A: forward={forward:.2f}, backward={backward:.2f}")
    print(f"  Expected: forward=0.0, backward=0.0")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    test_implementations()
