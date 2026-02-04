"""
Take each table in datalake and break them into singleton tables:
Each table with n columns becomes n singleton tables, one for each column. The new tables are renamed "column_name_table_name.csv".
Save the new tables in datalake/singletons.

"""
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
def break_into_singletons(datalake_dir):
    """
    Break each table in the datalake into singleton tables and rename the column in the table to "title".
    """
    for file in tqdm(os.listdir(datalake_dir)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(datalake_dir, file))
            for column in df.columns:
                new_df = df[[column]]
                new_df.columns = ["title"]
                new_df.to_csv(os.path.join("freyja-semantic-join/datalake/singletons/", f"{column.replace('/','_')}_{file}"), index=False)

def main():
    datalake_dir = "freyja-semantic-join/datalake"
    break_into_singletons(datalake_dir)

if __name__ == "__main__":
    main()