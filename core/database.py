import polars as pl
import os
import sqlite3

data_dir = "data"
sql_conn_str = "sqlite://"
mt_db = 'SQLite_FIADB_MT.db'


def get_df_from_db(table: str, selections):
    if not isinstance(selections, str):
        selections = ",".join(selections)
    db_path = os.path.join(data_dir, mt_db)
    conn = sqlite3.connect(db_path)
    df = pl.read_database(query=("SELECT " + selections +  " FROM " + table), connection=conn, infer_schema_length=None)
    return df