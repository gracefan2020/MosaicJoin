#!/usr/bin/env python3
"""
Test script to verify the Freyja semantic join analysis notebook setup
"""

import pandas as pd
import os

def test_data_loading():
    """Test that all required data files can be loaded"""
    print("Testing data loading...")
    
    # Check if files exist
    datalake_dir = "./freyja-semantic-join/datalake/"
    ground_truth_file = "./freyja-semantic-join/freyja_ground_truth.csv"
    
    if not os.path.exists(datalake_dir):
        print(f"❌ Datalake directory not found: {datalake_dir}")
        return False
    
    if not os.path.exists(ground_truth_file):
        print(f"❌ Ground truth file not found: {ground_truth_file}")
        return False
    
    # Test loading our selected tables
    tables_to_test = [
        'world_country.csv',
        'movies.csv', 
        'countries_and_continents.csv',
        'netflix_titles.csv',
        'fifa_ranking.csv'
    ]
    
    print(f"Testing {len(tables_to_test)} tables...")
    
    for table_file in tables_to_test:
        table_path = os.path.join(datalake_dir, table_file)
        if not os.path.exists(table_path):
            print(f"❌ Table not found: {table_path}")
            return False
        
        try:
            df = pd.read_csv(table_path)
            print(f"✅ {table_file}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"❌ Error loading {table_file}: {e}")
            return False
    
    # Test ground truth loading
    try:
        gt_df = pd.read_csv(ground_truth_file)
        print(f"✅ Ground truth: {gt_df.shape[0]} semantic join pairs")
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return False
    
    print("\n🎉 All data files loaded successfully!")
    return True

def test_country_columns():
    """Test that country columns exist in our tables"""
    print("\nTesting country columns...")
    
    datalake_dir = "./freyja-semantic-join/datalake/"
    
    # Expected country columns
    country_columns = {
        'world_country': 'Name',
        'movies': 'country', 
        'countries_and_continents': 'name',
        'netflix_titles': 'country',
        'fifa_ranking': 'country_full'
    }
    
    for table_name, expected_col in country_columns.items():
        table_path = os.path.join(datalake_dir, f"{table_name}.csv")
        try:
            df = pd.read_csv(table_path)
            if expected_col in df.columns:
                unique_count = df[expected_col].nunique()
                print(f"✅ {table_name}.{expected_col}: {unique_count} unique values")
            else:
                print(f"❌ Column {expected_col} not found in {table_name}")
                print(f"   Available columns: {list(df.columns)}")
                return False
        except Exception as e:
            print(f"❌ Error testing {table_name}: {e}")
            return False
    
    print("\n🎉 All country columns found!")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("FREYJA SEMANTIC JOIN ANALYSIS - SETUP TEST")
    print("=" * 60)
    
    success = True
    success &= test_data_loading()
    success &= test_country_columns()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! The notebook setup is ready.")
        print("\nNext steps:")
        print("1. Open freyja_semantic_join_analysis.ipynb in Jupyter")
        print("2. Implement the jaccard_similarity() function")
        print("3. Implement the set_containment() function")
        print("4. Run the analysis and answer the questions")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
