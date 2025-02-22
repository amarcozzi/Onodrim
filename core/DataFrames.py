import polars as pl
import os
import numpy as np

data_dir = "./data"
cond_fname = "MT_COND.csv"
tree_fname = "MT_TREE.csv"
plot_fname = "MT_PLOT.csv"

cond_selected_columns = ["CN", "PLT_CN", "SLOPE", "ASPECT", "BALIVE", "LIVE_CANOPY_CVR_PCT"]
tree_selected_columns = ["CN", "PLT_CN", "DIA", "ACTUALHT","HT", "TPA_UNADJ"]
plot_selected_columns = ["CN", "PLOT", "ELEV", "LAT", "LON"]

def create_polars_dataframe():
    print("Creating Polars DataFrame")

    cond_df = pl.read_csv(os.path.join(data_dir, cond_fname), columns=cond_selected_columns)
    #cond_df = cond_df.drop_nulls()
    COND = cond_df.sort("PLT_CN")
    tree_df = pl.read_csv(os.path.join(data_dir, tree_fname), columns=tree_selected_columns)
    TREE = tree_df.sort("PLT_CN")
    plot_df = pl.read_csv(os.path.join(data_dir, plot_fname), columns=plot_selected_columns)
    #plot_df = plot_df.drop_nulls()
    plot_df = plot_df.rename({"CN": "PLT_CN"})
    PLOT = plot_df.sort("PLT_CN")

    # grab diameters before grouping
    # DIA = TREE.get_column("DIA")




    # group trees by plot cn and join with our plot group
    # In MT_CSV, ACTUALHT is generally empty or the same as HT
    TREEGRP = pl.sql(
        query='''
            SELECT TREE.PLT_CN, 
                    COUNT(TREE.CN) AS TREE_COUNT, 
                    SUM(TREE.TPA_UNADJ), 
                    SUM(POW(TREE.DIA,2)) AS DIA_SQR_SUM, 
                    MAX(TREE.HT) AS MAX_HT, 
                    AVG(TREE.HT) AS AVG_HT
            FROM TREE
            GROUP BY TREE.PLT_CN
        '''
    ).collect()


    CONDGRP = pl.sql(
        query='''
        SELECT 
        PLOT.PLT_CN,
        PLOT.ELEV, 
        PLOT.LAT, 
        PLOT.LON, 
        COND.SLOPE, 
        COND.ASPECT, 
        COND.LIVE_CANOPY_CVR_PCT, 
        COND.BALIVE
        FROM PLOT NATURAL LEFT JOIN COND
        '''
    ).collect()
    print(CONDGRP)

    #select CONDGRP PLT_CN for 40,000
    #select TREEGRP PLT_CN for 12,000
    FINAL = pl.sql(
        query='''
        SELECT DISTINCT
            TREEGRP.PLT_CN,
            TREEGRP.TREE_COUNT,
            TREEGRP.TPA_UNADJ,
            TREEGRP.MAX_HT,
            TREEGRP.AVG_HT,
            CONDGRP.ELEV,
            CONDGRP.LAT,
            CONDGRP.LON,
            CONDGRP.SLOPE,
            CONDGRP.ASPECT,
            CONDGRP.LIVE_CANOPY_CVR_PCT,
            CONDGRP.BALIVE
            FROM CONDGRP NATURAL LEFT JOIN TREEGRP
            
        '''
    ).collect()

    print(FINAL)

    FINAL = FINAL.unique(subset=["PLT_CN"])
    FINAL = FINAL.drop_nulls()

    print("Final DataFrame: ")
    print(FINAL)
    FINAL.write_csv(os.path.join(data_dir, "output.csv"), separator=",")
    return FINAL

def create_avg_polars_dataframe():
    print("Creating Polars DataFrame with Average Values")
    cond_df = pl.read_csv(os.path.join(data_dir, cond_fname), columns=cond_selected_columns)
    # cond_df = cond_df.drop_nulls()
    COND = cond_df.sort("PLT_CN")
    tree_df = pl.read_csv(os.path.join(data_dir, tree_fname), columns=tree_selected_columns)
    TREE = tree_df.sort("PLT_CN")
    plot_df = pl.read_csv(os.path.join(data_dir, plot_fname), columns=plot_selected_columns)
    PLOT = plot_df.sort("CN")

    # grab diameters before grouping
    # DIA = TREE.get_column("DIA")

    PLOTGRP = pl.sql(
        query='''
        SELECT PLOT.ELEV, 
            PLOT.LAT, 
            PLOT.LON, 
            COALESCE(PLOT.CN, TREE.PLT_CN) AS PLT_CN,
        FROM PLOT NATURAL LEFT JOIN TREE

        '''
    ).collect()
    #print(PLOTGRP)

    CONDGRP = pl.sql(
        query='''
        SELECT COND.BALIVE,
            COND.SLOPE, 
            COND.ASPECT, 
            PLOT.ELEV, 
            PLOT.LAT, 
            PLOT.LON, 
            COND.LIVE_CANOPY_CVR_PCT, 
            COALESCE(COND.PLT_CN, PLOTGRP.PLT_CN) AS PLT_CN,
        FROM COND NATURAL LEFT JOIN PLOTGRP
        '''
    ).collect()
    #print(CONDGRP)

    # group trees by plot cn and join with our plot group
    # In MT_CSV, ACTUALHT is generally empty or the same as HT
    TREEGRP = pl.sql(
        query='''
            SELECT TREE.PLT_CN, 
                    COUNT(TREE.CN) AS TREE_COUNT, 
                    AVG(COND.LIVE_CANOPY_CVR_PCT),
                    AVG(COND.BALIVE),
                    SUM(TREE.TPA_UNADJ), 
                    MAX(TREE.HT) AS MAX_HT, 
                    AVG(TREE.HT) AS AVG_HT, 
                    AVG(PLOT.ELEV),
                    AVG(COND.SLOPE),
                    AVG(COND.ASPECT), 
                    AVG(PLOT.LAT), 
                    AVG(PLOT.LON), 
                    COALESCE(PLOT.CN, TREE.PLT_CN) AS PLT_CN
            FROM TREE NATURAL LEFT JOIN CONDGRP
            GROUP BY TREE.PLT_CN
        '''
    ).collect()


    print("Final DataFrame: ")
    print(TREEGRP)
    return TREEGRP


def main():
    print("main function")
    create_polars_dataframe()



if __name__ == "__main__":
    main()


