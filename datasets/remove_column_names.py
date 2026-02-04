import os
import pandas as pd

def remove_column_names(datalake_dir):
    """
    Replace each column name from each table in the datalake with "Title".
    Save the new copies of data to "datalake_no_column_names".

    """
    for file in os.listdir(datalake_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(datalake_dir, file))
            df.columns = ["title"]
            df.to_csv(os.path.join("wt/datalake_no_column_names", file), index=False)    


def remove_column_names_from_query_columns(query_columns_file):
    """
    Replace all values of "target_attr" with "title" from the query columns file.
    Save the new copies of data to "query_columns_no_column_names.csv".
    """
    df = pd.read_csv(query_columns_file)
    df["target_attr"] = "title"
    df.to_csv(query_columns_file.replace(".csv", "_no_column_names.csv"), index=False)


def remove_column_names_from_groundtruth(groundtruth_file):
    """
    Replace all values in "source_column" and "target_column" columns to "title" from the groundtruth file.
    Save the new copies of data to "groundtruth_no_column_names.csv".
    """
    df = pd.read_csv(groundtruth_file)
    for i, row in df.iterrows():
        # row["source_column"] = "title"
        # row["target_column"] = "title"
        row["target_ds"] = row["target_attr"]+"_"+row["target_ds"]
        row["candidate_ds"] = row["candidate_attr"]+"_"+row["candidate_ds"]
        row["target_attr"] = "title"
        row["candidate_attr"] = "title"
    df.to_csv(groundtruth_file.replace(".csv", "_no_column_names.csv"), index=False)

if __name__ == "__main__":
    # remove_column_names("wt/datalake")
    remove_column_names_from_query_columns("freyja-semantic-join/freyja_single_query_columns.csv")
    remove_column_names_from_groundtruth("freyja-semantic-join/freyja_ground_truth.csv")