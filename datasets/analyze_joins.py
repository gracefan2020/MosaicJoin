import pandas as pd
import os
import matplotlib.pyplot as plt

# Configuration: Set bucket size (5 for 5% buckets, 10 for 10% buckets, etc.)
BUCKET_SIZE = 5  # Change this to 10 for 10% buckets
DIR = "./santos-small"
GROUND_TRUTH_FILE = f"{DIR}/santos_small_ground_truth.csv"
DATALAKE_DIR = f"{DIR}/datalake/"
QUERY_DIR = f"{DIR}/query/"
# import the ground truth
gt_df = pd.read_csv(GROUND_TRUTH_FILE)

# import datalake


def value_overlap(seriesA, seriesB):
    # calculate the Jaccard similarity between two pandas Series
    # normalize the values, remove any special characters like spaces, commas, etc. (e.g., for value ' usa' -> 'usa')
    seriesA = seriesA.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    seriesB = seriesB.str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
    
    # Convert to sets to get unique values (immune to deduplication)
    setA = set(seriesA.dropna())
    setB = set(seriesB.dropna())
    
    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = setA.intersection(setB)
    union = setA.union(setB)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)



def plot_overlap_distribution(bucket_percentage_overlap, total_joins):
    # Sort buckets by lower bound of range
    sorted_buckets = sorted(bucket_percentage_overlap.keys(), key=lambda x: int(x.split('-')[0]))
    counts = [bucket_percentage_overlap[bucket] for bucket in sorted_buckets]
    percents = [(count / total_joins) * 100 for count in counts]
    # sanity check: sum of percents should be 100
    print(f"Sum of percents: {sum(percents)}")
    assert sum(percents) == 100
    # Plot bar chart
    plt.figure(figsize=(14,6))  # Wider figure for more space
    bar_width = 0.6  # Make bars narrower
    x = range(len(sorted_buckets))
    bars = plt.bar(x, percents, width=bar_width)
    plt.xlabel('Value Overlap (%) (Jaccard Similarity)')
    plt.ylabel('Percentage of Joinable Tables (%)')
    plt.title('Distribution of Value Overlap in Semantic Joins \n Freyja Benchmark')
    plt.xticks(x, sorted_buckets, rotation=45, ha='right')
    plt.tight_layout()
    # Add percentage numbers on top of each bar
    for bar, percent in zip(bars, percents):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{percent:.2f}%', 
            ha='center', 
            va='bottom',
            fontsize=10
        )
    # save the plot
    plt.savefig(f'{DIR}/join_analyses/overlap_distribution_freyja_{BUCKET_SIZE}pct.png')


num_no_overlap = 0
num_with_overlap = 0
num_no_overlap_distinct = 0
bucket_percentage_overlap = {}

# print("Columns with no overlapping values: ")
for index, row in gt_df.iterrows():
    query_ds = row["target_ds"]
    query_col = row["target_attr"]
    candidate_ds = row["candidate_ds"]
    candidate_col = row["candidate_attr"]
    # check if the join is in the datalake
    if os.path.exists(os.path.join(DATALAKE_DIR, query_ds)) and os.path.exists(os.path.join(DATALAKE_DIR, candidate_ds)):
        query_df = pd.read_csv(os.path.join(DATALAKE_DIR, query_ds))
        candidate_df = pd.read_csv(os.path.join(DATALAKE_DIR, candidate_ds))
        if query_col not in query_df.columns:   
            print(f"Column {query_col} not found in QUERY TABLE: {query_ds}")
            continue
        if candidate_col not in candidate_df.columns:
            print(f"Column {candidate_col} not found in CANDIDATE TABLE: {candidate_ds}")
            continue
        jaccard_similarity = value_overlap(query_df[query_col], candidate_df[candidate_col])

        if jaccard_similarity == 0:
            print(f"{query_ds} {query_col} \t {candidate_ds} {candidate_col} has no value overlap")
            # print(f"\t query values: {query_df[query_col].unique().tolist()}")
            # print(f"\t candidate values: {candidate_df[candidate_col].unique().tolist()}")
            num_no_overlap += 1
        else:
            num_with_overlap += 1
        percent_overlap = int(jaccard_similarity * 100)
        # Cap percent_overlap at 100 to prevent buckets above 100%
        percent_overlap = min(percent_overlap, 100)
        
        # Bucket the percent_overlap into configurable ranges
        bucket = (percent_overlap // BUCKET_SIZE) * BUCKET_SIZE
        # Merge the last bucket to include 100%
        max_bucket = 100 - BUCKET_SIZE
        if percent_overlap >= max_bucket:
            bucket_label = f"{max_bucket}-100%"
        else:
            bucket_label = f"{bucket}-{bucket+BUCKET_SIZE}%"
        if bucket_label not in bucket_percentage_overlap:
            bucket_percentage_overlap[bucket_label] = 0
        bucket_percentage_overlap[bucket_label] += 1
print(f"Number of semantic joins with no value overlap: {num_no_overlap} ({(num_no_overlap/len(gt_df))*100:.2f}%)")
print(f"Number of semantic joins with value overlap: {num_with_overlap} ({(num_with_overlap/len(gt_df))*100:.2f}%)")
print(f"Total number of semantic joins processed: {len(gt_df)}")
print(f"Total number of semantic joins in ground truth: {len(gt_df)}")

# Print buckets in ascending order and percentage of tables with this overlap
total_joins = len(gt_df)
# plot the overlap distribution
plot_overlap_distribution(bucket_percentage_overlap, total_joins)