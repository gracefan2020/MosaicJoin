import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration
DATASET_DIR = "./freyja-semantic-join"
DATALAKE_DIR = f"{DATASET_DIR}/datalake/"

def analyze_table_sizes():
    """Analyze the size distribution of all tables in the freyja-semantic-join dataset"""
    
    # Get all CSV files in the datalake
    csv_files = [f for f in os.listdir(DATALAKE_DIR) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files in {DATALAKE_DIR}")
    
    row_counts = []
    file_info = []
    error_files = []
    
    # Process each CSV file
    for i, filename in enumerate(csv_files):
        file_path = os.path.join(DATALAKE_DIR, filename)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            row_count = len(df)
            col_count = len(df.columns)
            
            row_counts.append(row_count)
            file_info.append({
                'filename': filename,
                'rows': row_count,
                'columns': col_count,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024)
            })
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(csv_files)} files...")
                
        except Exception as e:
            error_files.append({'filename': filename, 'error': str(e)})
            print(f"Error reading {filename}: {e}")
    
    # Calculate statistics
    row_counts_array = np.array(row_counts)
    
    stats = {
        'total_files': len(csv_files),
        'successful_files': len(row_counts),
        'error_files': len(error_files),
        'total_rows': np.sum(row_counts_array),
        'avg_rows': np.mean(row_counts_array),
        'median_rows': np.median(row_counts_array),
        'min_rows': np.min(row_counts_array),
        'max_rows': np.max(row_counts_array),
        'std_rows': np.std(row_counts_array),
        'q25_rows': np.percentile(row_counts_array, 25),
        'q75_rows': np.percentile(row_counts_array, 75)
    }
    
    # Print comprehensive statistics
    print("\n" + "="*60)
    print("FREYJA-SEMANTIC-JOIN DATASET ANALYSIS")
    print("="*60)
    
    print(f"\nFile Statistics:")
    print(f"  Total CSV files: {stats['total_files']}")
    print(f"  Successfully processed: {stats['successful_files']}")
    print(f"  Files with errors: {stats['error_files']}")
    
    print(f"\nRow Count Statistics:")
    print(f"  Total rows across all tables: {stats['total_rows']:,}")
    print(f"  Average rows per table: {stats['avg_rows']:.2f}")
    print(f"  Median rows per table: {stats['median_rows']:.2f}")
    print(f"  Minimum rows: {stats['min_rows']:,}")
    print(f"  Maximum rows: {stats['max_rows']:,}")
    print(f"  Standard deviation: {stats['std_rows']:.2f}")
    print(f"  25th percentile: {stats['q25_rows']:.2f}")
    print(f"  75th percentile: {stats['q75_rows']:.2f}")
    
    # Show distribution by row count ranges
    print(f"\nDistribution by Row Count Ranges:")
    ranges = [
        (0, 10, "1-10 rows"),
        (11, 50, "11-50 rows"),
        (51, 100, "51-100 rows"),
        (101, 500, "101-500 rows"),
        (501, 1000, "501-1,000 rows"),
        (1001, 5000, "1,001-5,000 rows"),
        (5001, 10000, "5,001-10,000 rows"),
        (10001, float('inf'), "10,001+ rows")
    ]
    
    for min_rows, max_rows, label in ranges:
        if max_rows == float('inf'):
            count = np.sum(row_counts_array >= min_rows)
        else:
            count = np.sum((row_counts_array >= min_rows) & (row_counts_array <= max_rows))
        percentage = (count / len(row_counts_array)) * 100
        print(f"  {label}: {count} tables ({percentage:.1f}%)")
    
    # Show largest and smallest tables
    file_info_df = pd.DataFrame(file_info)
    file_info_df = file_info_df.sort_values('rows', ascending=False)
    
    print(f"\nLargest Tables (by row count):")
    for _, row in file_info_df.head(10).iterrows():
        print(f"  {row['filename']}: {row['rows']:,} rows, {row['columns']} columns")
    
    print(f"\nSmallest Tables (by row count):")
    for _, row in file_info_df.tail(10).iterrows():
        print(f"  {row['filename']}: {row['rows']:,} rows, {row['columns']} columns")
    
    # Create visualizations
    create_distribution_plots(row_counts_array, stats)
    
    # Show error files if any
    if error_files:
        print(f"\nFiles with errors:")
        for error in error_files[:10]:  # Show first 10 errors
            print(f"  {error['filename']}: {error['error']}")
        if len(error_files) > 10:
            print(f"  ... and {len(error_files) - 10} more errors")
    
    return stats, file_info_df

def create_distribution_plots(row_counts, stats):
    """Create visualization plots for row count distribution"""
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram of row counts
    ax1.hist(row_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Rows')
    ax1.set_ylabel('Number of Tables')
    ax1.set_title('Distribution of Row Counts (All Tables)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2.boxplot(row_counts, vert=True)
    ax2.set_ylabel('Number of Rows')
    ax2.set_title('Box Plot of Row Counts')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_counts = np.sort(row_counts)
    cumulative_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100
    ax3.plot(sorted_counts, cumulative_pct, linewidth=2, color='green')
    ax3.set_xlabel('Number of Rows')
    ax3.set_ylabel('Cumulative Percentage')
    ax3.set_title('Cumulative Distribution of Row Counts')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar chart of ranges
    ranges = [
        (0, 10, "1-10"),
        (11, 50, "11-50"),
        (51, 100, "51-100"),
        (101, 500, "101-500"),
        (501, 1000, "501-1K"),
        (1001, 5000, "1K-5K"),
        (5001, 10000, "5K-10K"),
        (10001, float('inf'), "10K+")
    ]
    
    range_counts = []
    range_labels = []
    
    for min_rows, max_rows, label in ranges:
        if max_rows == float('inf'):
            count = np.sum(row_counts >= min_rows)
        else:
            count = np.sum((row_counts >= min_rows) & (row_counts <= max_rows))
        range_counts.append(count)
        range_labels.append(label)
    
    bars = ax4.bar(range_labels, range_counts, color='lightcoral', alpha=0.7)
    ax4.set_xlabel('Row Count Ranges')
    ax4.set_ylabel('Number of Tables')
    ax4.set_title('Distribution by Row Count Ranges')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, range_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(range_counts)*0.01,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = f"{DATASET_DIR}/table_analyses"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/row_count_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\nDistribution plots saved to: {output_dir}/row_count_distribution.png")
    
    plt.show()

if __name__ == "__main__":
    stats, file_info_df = analyze_table_sizes()
