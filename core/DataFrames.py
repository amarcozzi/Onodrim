import polars as pl
import os
import json
import numpy as np

import database as db

sql_conn_str = "sqlite://"
mt_db = 'SQLite_FIADB_MT.db'
fortyp_json = "forest_type_codes.json"
data_dir = "data"


cond_selected_columns = ["PLT_CN", "ASPECT", "BALIVE", "LIVE_CANOPY_CVR_PCT", "FORTYPCD",]
subp_selected_columns = ["SUBP", "PLT_CN", "SLOPE", "ASPECT"]
tree_selected_columns = ["CN","SUBP", "PLT_CN", "STATUSCD", "DIA", "ACTUALHT","HT", "TPA_UNADJ"]
plot_selected_columns = ["CN", "PLOT", "DESIGNCD", "ELEV", "LAT", "PLOT_STATUS_CD"]

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
    SUBP = SUBP.sort("PLT_CN")
    #print(SUBP)

    SUBPGRP= SUBP.join(TREE, on="PLT_CN", how="left")
    TEST = pl.sql(
        query='''
            SELECT PLOT.PLOT_STATUS_CD,
                    COUNT(PLOT.PLT_CN) AS COUNT
            FROM PLOT
            GROUP BY PLOT_STATUS_CD
        '''
    ).collect()

    print(TEST)
    #print(SUBPGRP)
    # group trees by plot cn AND by subplot
    # In MT_CSV, ACTUALHT is generally empty or the same as HT
    # This SQL query is done this way for readability and future potential adjustments
    TREEGRP = pl.sql(
        query='''
            SELECT TREE.PLT_CN, TREE.SUBP, 
                    COUNT(TREE.CN) AS TREE_COUNT, 
                    SUM(TREE.TPA_UNADJ), 
                    SUM(POW(TREE.DIA,2)) AS DIA_SQR_SUM, 
                    MAX(TREE.HT) AS MAX_HT, 
                    AVG(TREE.HT) AS AVG_HT
            FROM TREE
            WHERE TREE.STATUSCD = 1
            GROUP BY TREE.PLT_CN, TREE.SUBP
            ORDER BY TREE.PLT_CN, TREE.SUBP
        '''
    ).collect()

    print(TREEGRP)

    #join out PLOT and COND tables together
    CONDGRP = PLOT.join(COND,on="PLT_CN", how="left")
    CONDGRP = CONDGRP.drop_nulls()
    print(f"CONDGRP with dropped nulls {CONDGRP}")


    #join our CONDGRP with our TREEGRP tables for final data frame
    FINAL = TREEGRP.join(CONDGRP, on="PLT_CN", how="inner")
    FINAL = FINAL.with_columns([
        np.cos(np.radians(FINAL['ASPECT'])).alias("ASPECT_COS"),
        np.sin(np.radians(FINAL['ASPECT'])).alias("ASPECT_SIN"),
    ])

    print(f"Final DataFrame {FINAL} ")
    FINAL.write_csv(os.path.join(data_dir, "output.csv"), separator=",") #write as csv for checks and records

    #test our filtering by giving it a real code and a fake code
    pondo = filter_by_forest_type(FINAL, 221)
    error = filter_by_forest_type(FINAL, 219)

    return FINAL

def filter_by_forest_type(dataframe: pl.DataFrame, fortypcd: int):
    print(f"Filtering our data by forest type code {fortypcd}")
    if is_forest_type(fortypcd):
        return dataframe.filter(pl.col("FORTYPCD") == fortypcd) #if it exists, return our filtered dataframe
    else:
        return dataframe #else return unaltered dataframe


def is_forest_type(fortypcd: int):
    with open(os.path.join(data_dir, fortyp_json)) as json_file:
        typcds = json.load(json_file)
    values = list(typcds.values())
    for d in values:
        if str(fortypcd) in d.keys():
            print(f"Given forest type code {fortypcd} found")
            return True

    print(f"Given forest type code {fortypcd} does not exist")
    return False

def main():
    print("DataGrames.py main function")
    create_polars_dataframe()



if __name__ == "__main__":
    main()


