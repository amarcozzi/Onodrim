import polars as pl
import os
import json
import numpy as np

import database as db

sql_conn_str = "sqlite://"
mt_db = 'SQLite_FIADB_MT.db'
fortyp_json = "forest_type_codes.json"
data_dir = "data"


cond_selected_columns = ["CN", "PLT_CN", "ASPECT", "BALIVE", "LIVE_CANOPY_CVR_PCT", "FORTYPCD", "STDAGE", "QMD_RMRS"]
subp_selected_columns = ["CN", "PLT_CN", "SLOPE", "ASPECT"]
tree_selected_columns = ["CN", "PLT_CN", "STATUSCD", "DIA", "ACTUALHT","HT", "TPA_UNADJ"]
plot_selected_columns = ["CN", "PLOT", "ELEV", "LAT"]

def create_polars_dataframe():
    print("Creating Polars DataFrame")

    #retrieve our data from the SQL database
    COND = db.get_df_from_db("COND", cond_selected_columns)
    COND = COND.sort("PLT_CN")

    TREE = db.get_df_from_db("TREE", tree_selected_columns)
    TREE = TREE.sort("PLT_CN")

    PLOT = db.get_df_from_db("PLOT", plot_selected_columns)
    PLOT = PLOT.rename({"CN": "PLT_CN"})
    PLOT = PLOT.sort("PLT_CN")

    SUBP = db.get_df_from_db("SUBPLOT", subp_selected_columns)
    SUBP = SUBP.drop_nulls()
    print(SUBP)


    # group trees by plot cn and join with our plot group
    # In MT_CSV, ACTUALHT is generally empty or the same as HT
    # This SQL query is done this way for readability and future potential adjustments
    TREEGRP = pl.sql(
        query='''
            SELECT TREE.PLT_CN, 
                    COUNT(TREE.CN) AS TREE_COUNT, 
                    SUM(TREE.TPA_UNADJ), 
                    SUM(POW(TREE.DIA,2)) AS DIA_SQR_SUM, 
                    MAX(TREE.HT) AS MAX_HT, 
                    AVG(TREE.HT) AS AVG_HT
            FROM TREE
            WHERE TREE.STATUSCD = 1
            GROUP BY TREE.PLT_CN
        '''
    ).collect()

    #join out PLOT and COND tables together
    CONDGRP = PLOT.join(COND,on="PLT_CN", how="left")
    print(CONDGRP)

    #join our CONDGRP with our TREEGRP tables for final data frame
    FINAL = CONDGRP.join(TREEGRP, on="PLT_CN", how="left")

    FINAL = FINAL.with_columns([
        np.cos(np.radians(FINAL['ASPECT'])).alias("ASPECT_COS"),
        np.sin(np.radians(FINAL['ASPECT'])).alias("ASPECT_SIN"),
    ])

    print(FINAL)

    #drop duplicates and nulls
    FINAL = FINAL.unique(subset=["PLT_CN"])
    FINAL = FINAL.drop_nulls()

    print("Final DataFrame: ")
    print(FINAL)
    FINAL.write_csv(os.path.join(data_dir, "output.csv"), separator=",") #write as csv for checks and records
    return FINAL

def filter_by_forest_type(dataframe: pl.DataFrame, fortypcd: int):
    print("Filtering our data by forest type code")
    with open(os.path.join(data_dir, fortyp_json)) as json_file:
        typcds = json.load(json_file)

    if fortypcd in typcds.values():
        return dataframe.filter(pl.col("FORTYPCD") == fortypcd)
    else:
        print("Given forest type code does not exist")
        return


def main():
    print("DataGrames.py main function")
    create_polars_dataframe()



if __name__ == "__main__":
    main()


