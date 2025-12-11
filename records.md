# 1. DeepJoin Prefiltering
## query_results_k1024_t0.7_top50_deepjoin_N100_K500_T0.6:
 DeepJoin (k=100) + 
Sketches for Semantic Join 
(sketch size = 1024, k=50)
Precision: 0.887
Recall: 0.691
F1-score: 0.777


## with LLM post-processing:
### Sample prompt: 
You are a helpful assistant that helps with semantic joinability analysis.
You are given a query table, query column, candidate table, and candidate column, and you need to determine if they are semantically joinable.
The query table, called {query_table_name}, has schema {query_schema}. The first 5 rows are:
{query_table_rows}
The query column is {query_column_name}, with values {query_col_vals}.
The candidate table, called {candidate_table_name}, has schema {candidate_schema}. The first 5 rows are:
{candidate_table_rows}
The candidate column is {candidate_column_name}, with values {candidate_col_vals}.
Determine if the query column is semantically joinable with the candidate column, based on their semantics and their value semantic overlap. 
Return "Yes" if they are semantically joinable, "No" otherwise.
### Results:
Precision: 0.930
Recall: 0.286
F1: 0.437


### Sample Prompt (specificing column semantics vs. table semantics):
You are a helpful assistant that helps with semantic joinability analysis.

TASK: Determine if two columns are semantically joinable.

Two columns are SEMANTICALLY JOINABLE if:
1. They represent the SAME CONCEPT (e.g., both are cities, both are person names, both are companies)
2. They have significant SEMANTIC OVERLAP in their values (the same entities appear in both columns, even if written differently. For example, "NYC", "New York City", "Big Apple" all represent the same entity)
3. The underlying TABLES can have DIFFERENT semantics (e.g., an airports table and a hospitals table can still be joinable on city columns)

DATA PROVIDED:
Query table "{query_table_name}" (schema: {query_schema})
First 5 rows:
{query_table_rows}
Query column "{query_column_name}" values: {query_col_vals}

Candidate table "{candidate_table_name}" (schema: {candidate_schema})
First 5 rows:
{candidate_table_rows}
Candidate column "{candidate_column_name}" values: {candidate_col_vals}

INSTRUCTIONS:
Analyze if the query column and candidate column represent the same semantic concept and if their values have substantial semantic overlap (same entities, possibly with different surface forms). Consider that the tables themselves may have different purposes.

Return ONLY "Yes" if they are semantically joinable, or "No" if they are not.

### Results:
Precision: 0.939
Recall: 0.584
F1: 0.720 


### Sample Prompt (adding 2 examples, taking 200 samples from sketch):
You are a helpful assistant that helps with semantic joinability analysis.

TASK: Determine if two columns are semantically joinable.

Two columns are JOINABLE if:
1. Same CONCEPT (e.g., cities, names, IDs)
2. Substantial VALUE OVERLAP (same entities, different forms: "NYC"="New York City"="Big Apple")
3. Tables may differ (airports vs hospitals joinable on city columns)

EXAMPLES:

Example 1:
Query: table="airports", column="city", schema=["airport_code", "city", "state"]
Values: ["New York City", "Los Angeles", "Chicago", "Houston", "Phoenix"]

Candidate: table="hospitals", column="location", schema=["hospital_name", "location", "beds"]
Values: ["NYC", "LA", "Chicago", "Houston", "Phoenix"]

Answer: Yes
Reason: Both represent cities/locations with substantial value overlap, even though formats differ ("New York City" vs "NYC", "Los Angeles" vs "LA").

Example 2:
Query: table="universities", column="university_name", schema=["university_name", "state", "enrollment"]
Values: ["MIT", "Stanford", "Harvard", "Caltech", "Princeton"]

Candidate: table="companies", column="company_name", schema=["company_name", "industry", "revenue"]
Values: ["Apple", "Google", "Microsoft", "Amazon", "Meta"]

Answer: No
Reason: Different concepts (university names vs company names) with no value overlap. These are different entity types, not just different table topics.



Query: table="{query_table_name}", column="{query_column_name}", schema={query_schema}
Values: {query_col_vals[:max_values]}

Candidate: table="{candidate_table_name}", column="{candidate_column_name}", schema={candidate_schema}
Values: {candidate_col_vals[:max_values]}

Return ONLY "Yes" if they are semantically joinable, or "No" if they are not.

### Results:
Precision: 0.951
Recall: 0.684
F1: 0.796 



### Sample Prompt (adding 2 examples, taking 200 samples from sketch):
You are a helpful assistant that helps with semantic joinability analysis.

TASK: Determine if these columns are semantically joinable for feature augmentation in machine learning. The goal is to identify columns that refer to the same real-world entities, even if they use different syntactic forms or have partial overlap. When in doubt, err on the side of inclusion - these joins will provide additional features that can help ML models.

Two columns are SEMANTICALLY JOINABLE if they represent the SAME TYPE OF ENTITY or CONCEPT, even if:
- Values have different syntactic forms (e.g., "NYC" vs "New York City", "USA" vs "United States")
- There is partial (not complete) value overlap, but more than half of the values have a semantic overlap/
- Formats differ (abbreviations, full names, codes, etc.)

EXAMPLES:

Example 1:
Query: table="airports", column="city", schema=["airport_code", "city", "state"]
Values: ["New York City", "Los Angeles", "Chicago", "Houston", "Phoenix"]

Candidate: table="hospitals", column="location", schema=["hospital_name", "location", "beds"]
Values: ["NYC", "LA", "Chicago", "Houston", "Phoenix"]

Answer: Yes
Reason: Both represent cities/locations with clear semantic overlap, even though formats differ ("New York City" vs "NYC", "Los Angeles" vs "LA"). Different syntactic forms are expected and acceptable.

Example 2:
Query: table="people", column="name", schema=["name", "date_of_birth", "gender"]
Values: ["Apple", "Dell", "Tiffany"]

Candidate: table="companies", column="name", schema=["name", "industry", "revenue"]
Values: ["Apple", "Google", "Ford", "Dell", "Tiffany"]

Answer: No
Reason: Completely different entity types (people names vs company names) with no semantic relationship or value overlap. These represent fundamentally different concepts.

---

Query: table="{query_table_name}", column="{query_column_name}", schema={query_schema}
Values: {query_col_vals[:max_values]}

Candidate: table="{candidate_table_name}", column="{candidate_column_name}", schema={candidate_schema}
Values: {candidate_col_vals[:max_values]}

Return "Yes" if semantically joinable, "No" otherwise."""


### Results:
Precision: 0.927
Recall: 0.729
F1: 0.816