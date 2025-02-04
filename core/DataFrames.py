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




def main():
    print("main function")



if __name__ == "__main__":
    main()

def create_polars_dataframe():
    print("Creating Polars DataFrame")

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
        SELECT PLOT.ELEV, PLOT.LAT, PLOT.LON, COALESCE(PLOT.CN, TREE.PLT_CN) AS PLT_CN,
        FROM PLOT NATURAL LEFT JOIN TREE

        '''
    ).collect()
    #print(PLOTGRP)

    CONDGRP = pl.sql(
        query='''
        SELECT COND.SLOPE, COND.ASPECT, PLOT.ELEV, PLOT.LAT, PLOT.LON, COND.LIVE_CANOPY_CVR_PCT, COALESCE(COND.PLT_CN, PLOTGRP.PLT_CN) AS PLT_CN,
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
                    SUM(TREE.TPA_UNADJ), 
                    SUM(POW(TREE.DIA,2)) AS DIA_SQR_SUM, 
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

    # calculate QMD from our data
    DIAS = TREEGRP.get_column("DIA_SQR_SUM")
    CNCOUNT = TREEGRP.get_column("TREE_COUNT")
    QMD = np.sqrt(DIAS / CNCOUNT)
    QMD = QMD.alias("QMD")
    TREEGRP.insert_column(4, QMD)

    print("Final DataFrame: ")
    print(TREEGRP)
    return TREEGRP
