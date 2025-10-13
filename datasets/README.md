# Freyja Semantic Join Analysis - Lecture Notebook

This repository contains a Jupyter notebook for teaching semantic joins using the Freyja benchmark dataset.

## Overview

The notebook `freyja_semantic_join_analysis.ipynb` provides a hands-on learning experience for students to:

1. **Understand semantic joins** and their applications in data integration
2. **Implement similarity measures** including Jaccard similarity and set containment
3. **Analyze real-world data** using 5 interesting tables from the Freyja benchmark
4. **Evaluate results** against ground truth data
5. **Visualize relationships** between different datasets

## Files

- `freyja_semantic_join_analysis.ipynb` - Main lecture notebook (stencil code for students)
- `sample_solution.py` - Sample implementation for instructors
- `test_notebook_setup.py` - Test script to verify data loading
- `freyja-semantic-join/` - Freyja benchmark dataset
  - `datalake/` - Contains 160 CSV files with various datasets
  - `freyja_ground_truth.csv` - Ground truth semantic join pairs
- `analyze_joins.py` - Original analysis script (reference)

## Selected Tables

The notebook focuses on 6 strategically chosen tables that demonstrate key concepts:

1. **world_country.csv** - Comprehensive country information (239 countries) - Large, comprehensive reference
2. **countries_and_continents.csv** - Detailed country metadata (251 countries) - Similar to world_country
3. **USA_cars_datasets.csv** - Cars from USA/Canada only (2,499 cars) - **Demonstrates Jaccard bias!**
4. **language.csv** - Languages with 2-letter country codes (2,679 languages) - **Format mismatch!**
5. **world-cities.csv** - Cities with countries (23,018 cities) - **Very large dataset**
6. **sloganlist.csv** - Company slogans (1,162 companies) - **No country column - false positive!**

## Learning Objectives

### Core Concepts
- **Semantic Joins**: Joining tables based on meaning rather than exact matches
- **Jaccard Similarity**: Measuring overlap between sets of values
- **Set Containment**: Measuring how much one set is contained in another
- **Data Preprocessing**: Normalizing text data for better matching

### Technical Skills
- Implementing similarity functions from scratch
- Working with pandas Series and data manipulation
- Creating visualizations with matplotlib and seaborn
- Evaluating algorithms against ground truth

## Setup Instructions

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required packages: pandas, numpy, matplotlib, seaborn

### Installation
```bash
# Activate your conda environment
conda activate profile

# Install required packages (if not already installed)
pip install pandas numpy matplotlib seaborn jupyter

# Verify setup
python test_notebook_setup.py
```

### Running the Notebook
1. Start Jupyter Notebook: `jupyter notebook`
2. Open `freyja_semantic_join_analysis.ipynb`
3. Follow the instructions to implement the required functions
4. Run the analysis and answer the questions

## Student Tasks

### Primary Tasks
1. **Implement `jaccard_similarity()`** - Calculate Jaccard similarity between two pandas Series
2. **Implement `set_containment()`** - Calculate forward and backward containment
3. **Analyze semantic relationships** between the 5 selected tables
4. **Visualize results** using heatmaps and scatter plots
5. **Evaluate against ground truth** to validate implementations

### Analysis Questions
Students are asked to answer:
1. **Jaccard Bias Demonstration**: Compare usa_cars.country with world_country.Name
2. **False Positives**: What happens with sloganlist and language tables?
3. **Size Effects**: How does table size affect similarity measures?
4. **Format Mismatches**: How to handle "USA" vs "US" vs "United States"?
5. **Practical Implications**: When to use each similarity measure?

### Advanced Tasks (Optional)
1. Implement weighted Jaccard similarity
2. Create a semantic join recommendation system
3. Analyze the impact of data preprocessing
4. Compare different normalization strategies
5. Implement fuzzy matching

## Expected Results

### High Similarity Pairs
- `world_country.Name` ↔ `countries_and_continents.name` (very high similarity)
- `world_cities.country` ↔ `world_country.Name` (moderate-high similarity)

### Key Insights
- **Jaccard Bias**: usa_cars vs world_country shows misleadingly low similarity
- **Format Mismatches**: language.countrycodes vs world_country.Name (2-letter vs full names)
- **Size Effects**: Large tables like world_cities affect similarity scores
- **False Positives**: sloganlist has no country data but might appear related
- **Containment vs Similarity**: Forward containment better for subset relationships

## Instructor Notes

### Teaching Tips
1. **Start with examples**: Use simple examples to explain Jaccard similarity
2. **Emphasize preprocessing**: Show how normalization affects results
3. **Discuss edge cases**: Empty sets, missing values, special characters
4. **Connect to real applications**: Data integration, record linkage, etc.

### Common Student Challenges
1. **Handling missing values**: Students often forget to drop NaN values
2. **Normalization**: Inconsistent text preprocessing
3. **Edge cases**: Division by zero, empty sets
4. **Understanding measures**: Confusion between similarity and containment

### Assessment
- **Code implementation**: Correctness of similarity functions
- **Analysis quality**: Thoughtful answers to questions
- **Understanding**: Ability to explain differences between measures
- **Creativity**: Optional advanced implementations

## Dataset Information

The Freyja benchmark contains:
- **160 CSV files** with diverse datasets
- **1,528 ground truth pairs** of semantic joins
- **Multiple domains**: Countries, movies, books, financial data, etc.
- **Real-world complexity**: Inconsistent formatting, missing values, typos

## References

- Freyja Benchmark: [Paper/Repository Link]
- Semantic Joins: [Relevant Papers]
- Jaccard Similarity: [Wikipedia/Textbook References]

## License

This educational material is provided for academic use. Please cite appropriately if used in research or publications.