import polars as pl
import os
import sqlite3

data_dir = "data"
sql_conn_str = "sqlite://"
mt_db = 'SQLite_FIADB_MT.db'
db = "SQLite_FIADB_"


def get_df_from_db(state: str, table: str, selections):

    #if we have an array, join our selected columns into a string for query
    if not isinstance(selections, str):
        selections = ",".join(selections)

    db_name = db + state + ".db"
    db_path = os.path.join(data_dir, db_name)
    conn = sqlite3.connect(db_path) #connect to SQL database
    df = pl.read_database(query=("SELECT " + selections +  " FROM " + table), connection=conn, infer_schema_length=None)
    return df