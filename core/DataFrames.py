import sys

import polars as pl
import os
import json
import csv
import numpy as np
import database as db
import rioxarray
from pyproj import Transformer

sql_conn_str = "sqlite://"
fortyp_json = "forest_type_codes.json"
data_dir = "data"

#FIADB desired columns
cond_selected_columns = ["PLT_CN", "ASPECT", "SLOPE", "FORTYPCD"]
subp_selected_columns = ["PLT_CN", "SUBP"]
tree_selected_columns = ["CN","SUBP", "PLT_CN", "STATUSCD", "DIA", "ACTUALHT","HT", "TPA_UNADJ"]
plot_selected_columns = ["CN", "DESIGNCD", "ELEV", "LAT", "LON", "PLOT_STATUS_CD"]

#Bioclimatic data file paths
bioclim_fnames = [
    "wc2.1_10m_bio_1.tif",
    "wc2.1_10m_bio_2.tif",
    "wc2.1_10m_bio_3.tif",
    "wc2.1_10m_bio_4.tif",
    "wc2.1_10m_bio_5.tif",
    "wc2.1_10m_bio_6.tif",
    "wc2.1_10m_bio_7.tif",
    "wc2.1_10m_bio_8.tif",
    "wc2.1_10m_bio_9.tif",
    "wc2.1_10m_bio_10.tif",
    "wc2.1_10m_bio_11.tif",
    "wc2.1_10m_bio_12.tif",
    "wc2.1_10m_bio_13.tif",
    "wc2.1_10m_bio_14.tif",
    "wc2.1_10m_bio_15.tif",
    "wc2.1_10m_bio_16.tif",
    "wc2.1_10m_bio_17.tif",
    "wc2.1_10m_bio_18.tif",
    "wc2.1_10m_bio_19.tif"
]
#LIST OF OUR NAMES FOR RENAMING WHEN READING CSV
bio_names = ["SUBPLOT_ID",              #CSV HAS SUBPLOT_ID
             "MEAN_TEMP",               #BIO1   ANNUAL MEAN TEMP
             "MEAN_DIURNAL_RANGE",      #BIO2   MEAN OF MONTHLY (MAX TEMP _ MIN TEMP)
             "ISOTHERMALITY",           #BIO3   (BIO2/BIO7)*100
             "TEMP_SEASONALITY",        #BIO4   (STD DEV * 100)
             "MAX_TEMP_WARM_MONTH",     #BIO5
             "MIN_TEMP_COLD_MONTH",     #BIO6
             "TEMP_RANGE",              #BIO7   (BIO5 - BIO6)
             "MEAN_TEMP_WET_QUARTER",   #BIO8
             "MEAN_TEMP_DRY_QUARTER",   #BIO9
             "MEAN_TEMP_WARM_QUARTER",  #BIO10
             "MEAN_TEMP_COLD_QUARTER",  #BIO11
             "ANNUAL_PRECIP",           #BIO12
             "PRECIP_WET_MONTH",        #BIO13
             "PRECIP_DRY_MONTH",        #BIO14
             "PRECIP_SEASONALITY",      #BIO15  (COEFFICIENT of VARIATION)
             "PRECIP_WET_QUARTER",      #BIO16
             "PRECIP_DRY_QUARTER",      #BIO17
             "PRECIP_WARM_QUARTER",     #BIO18
             "PRECIP_COLD_QUARTER"      #BIO19
             ]
clim_dir = os.path.join(data_dir, "climatic")

def create_polars_dataframe_by_subplot():
    print("Creating Polars DataFrame")

    #retrieve our data from the SQL database
    COND = db.get_df_from_db("MT","COND", cond_selected_columns)
    COND = COND.sort("PLT_CN")

    TREE = db.get_df_from_db("MT", "TREE", tree_selected_columns)
    TREE = TREE.sort("PLT_CN")

    PLOT = db.get_df_from_db("MT", "PLOT", plot_selected_columns)
    PLOT = PLOT.rename({"CN": "PLT_CN"})
    PLOT = PLOT.sort("PLT_CN")


    SUBP = db.get_df_from_db("MT", "SUBPLOT", subp_selected_columns)
    SUBP = SUBP.join(COND, on="PLT_CN", how="right", coalesce=True)
    SUBP = SUBP.drop_nulls()
    #create our subplot id for readability
    SUBP = SUBP.with_columns(
                                pl.concat_str(
                                [
                                        pl.col("PLT_CN"),
                                        pl.col("SUBP")
                                        ],
                                        separator= "_",
                                    ).alias("SUBPLOT_ID"),
                                )
    SUBP = SUBP.sort("SUBPLOT_ID")
    print(SUBP)

    #calculate the basal area of each stem recorded in the TREE table
    TREE = TREE.with_columns([
        (np.square(pl.col("DIA")) * 0.005454).alias("BASAL_AREA_STEM")
    ])

    # group trees by plot cn AND by subplot
    # In MT_CSV, ACTUALHT is generally empty or the same as HT
    # This SQL query is done this way for readability and future potential adjustments
    TREEGRP = pl.sql(
        query='''
            SELECT TREE.PLT_CN, TREE.SUBP, 
                    COUNT(TREE.CN) AS STEM_COUNT,
                    SUM(IF(TREE.DIA>=5.0,1,0)) AS TREE_COUNT,
                    SUM(IF(TREE.DIA<5.0,1,0)) AS SAPL_COUNT, 
                    SUM(IF(TREE.DIA>=5.0,TREE.BASAL_AREA_STEM,0)) AS BASAL_AREA_TREE, 
                    SUM(IF(TREE.DIA<5.0,TREE.BASAL_AREA_STEM,0)) AS BASAL_AREA_SAPL, 
                    SUM(TREE.BASAL_AREA_STEM) AS TOTAL_BASAL_AREA,
                    SUM(IF(TREE.DIA>=5.0,POW(TREE.DIA,2),0)) AS DIA_SQR_TREE, 
                    SUM(IF(TREE.DIA<5.0,POW(TREE.DIA,2),0)) AS DIA_SQR_SAPL,
                    MAX(TREE.HT) AS MAX_HT, 
                    AVG(TREE.HT) AS AVG_HT
            FROM TREE
            WHERE TREE.STATUSCD = 1
            GROUP BY TREE.PLT_CN, TREE.SUBP
            ORDER BY TREE.PLT_CN, TREE.SUBP
        '''
    ).collect()

    #create our SUBPLOT_ID key for readability
    TREEGRP = TREEGRP.with_columns(
                                pl.concat_str(
                                [
                                        pl.col("PLT_CN"),
                                        pl.col("SUBP")
                                        ],
                                        separator= "_",
                                    ).alias("SUBPLOT_ID"),
                                )
    #create our SUBPLOTID key for practical use
    #THIS KEY IS NECESSARY BECAUSE IT CAN BE CONVERTED TO A LONG FOR TRAINING
    TREEGRP = TREEGRP.with_columns(
                                pl.concat_str(
                                [
                                        pl.col("PLT_CN"),
                                        pl.col("SUBP")
                                        ],
                                        separator= "",
                                    ).alias("SUBPLOTID"),
                                )
    TREEGRP = TREEGRP.drop_nulls()
    TREEGRP = TREEGRP.unique("SUBPLOT_ID")
    TREEGRP = TREEGRP.sort("SUBPLOT_ID")

    # 1 acre = 43560 ft^2
    # subplot radius = 24 ft, subplot area = ~1809.56 ft^2
    TREEGRP = TREEGRP.with_columns([
        np.sqrt(pl.col("DIA_SQR_TREE") / (pl.col("TREE_COUNT"))).alias("QMD_TREE")
    ])

    print(TREEGRP)

    #join out PLOT, SUBP, and COND tables together
    CONDGRP = PLOT.join(SUBP,on="PLT_CN", how="right", coalesce=True)
    CONDGRP = CONDGRP.drop_nulls()
    CONDGRP = CONDGRP.unique("SUBPLOT_ID")
    CONDGRP = CONDGRP.sort("SUBPLOT_ID")
    print(f"CONDGRP with dropped nulls {CONDGRP}")


    #join our CONDGRP with our TREEGRP tables for final data frame
    FINAL = TREEGRP.join(CONDGRP, on=["SUBPLOT_ID","SUBP","PLT_CN"], coalesce=True)
    FINAL = FINAL.with_columns(pl.col("SUBPLOTID").cast(pl.Int64))
    FINAL = FINAL.with_columns([
        np.cos(np.radians(FINAL['ASPECT'])).alias("ASPECT_COS"),
        np.sin(np.radians(FINAL['ASPECT'])).alias("ASPECT_SIN")
    ])
    FINAL = FINAL.drop_nulls()
    FINAL = FINAL.sort("SUBPLOT_ID")

    #add our climate variables
    if not os.path.exists(os.path.join(data_dir, "climate_data.csv")):
        climate_variables_to_csv(FINAL.select(["SUBPLOT_ID", "LAT", "LON"]))

    #uses climate_data.csv to add climate data to FINAL
    FINAL = climate_variables_to_df(FINAL)

    print(f"Final DataFrame {FINAL} ")
    FINAL.write_csv(os.path.join(data_dir, "output.csv"), separator=",")  # write as csv for checks and records

    #test our filtering by giving it a real code and a fake code
    #pondo = filter_by_forest_type(FINAL, 221)
    #error = filter_by_forest_type(FINAL, 219)
    return FINAL


def climate_variables_to_csv(plots):
    print("Saving our climate variables to csv file: ")
    field_names = ["SUBPLOT_ID"]
    bioclim_data = [rioxarray.open_rasterio(os.path.join(clim_dir, f), parse_coordinates=True,default_name=get_field_name(f, field_names)) for f in bioclim_fnames]

    transformer = Transformer.from_crs("WGS 84", bioclim_data[0].rio.crs, always_xy=True)
    features = {}

    #open our csv
    with open(os.path.join(data_dir, "climate_data.csv"), 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        csv_writer.writeheader()
        total_rows = plots.height
        cur_row = 0
        for plot in plots.iter_rows(named=True):
            lat, lon = plot["LAT"], plot["LON"]
            xx, yy = transformer.transform(lon,lat)
            features["SUBPLOT_ID"] = plot["SUBPLOT_ID"]
            for data in bioclim_data:
                #get our data value
                value = data.sel(x=xx, y=yy, method='nearest').values
                features[data.name] = value[0]

            csv_writer.writerow(features)
            cur_row = cur_row + 1
            sys.stdout.write(f"\t\r{(cur_row/total_rows)*100} % complete")
            sys.stdout.flush()
    print("...Done")





def climate_variables_to_df(df):
    print(f"Assigning climate variables to given data frame: ")
    csv_df = pl.read_csv(os.path.join(data_dir, "climate_data.csv"), new_columns=bio_names)
    df = df.join(csv_df, on="SUBPLOT_ID", coalesce=True)
    return df


#function gets our climate variable field name and updates the list of names for processing
def get_field_name(fname, field_names):
    name = os.path.split(fname)[1] #split path into filename
    name = name.split(".")[1].split("_") #split filename into parts
    name = name[2] + name[3] #combine our two descriptive words to match worldclim coding
    field_names.append(name) #python allows lists to be mutable! saves us another for loop
    return name

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
    create_polars_dataframe_by_subplot()
    #create_polars_dataframe_by_plot()



if __name__ == "__main__":
    main()



#OUTDATED
#not updated with current metrics
def create_polars_dataframe_by_plot():
    print("Creating Polars DataFrame")

    #retrieve our data from the SQL database
    COND = db.get_df_from_db("MT","COND", cond_selected_columns)
    COND = COND.sort("PLT_CN")

    TREE = db.get_df_from_db("MT", "TREE", tree_selected_columns)
    TREE = TREE.sort("PLT_CN")

    PLOT = db.get_df_from_db("MT","PLOT", plot_selected_columns)
    PLOT = PLOT.rename({"CN": "PLT_CN"})
    PLOT = PLOT.sort("PLT_CN")


    # group trees by plot cn AND by subplot
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
            ORDER BY TREE.PLT_CN
        '''
    ).collect()

    TREEGRP = TREEGRP.unique("PLT_CN")
    TREEGRP = TREEGRP.with_columns([
        np.sqrt(pl.col("DIA_SQR_SUM") / (pl.col("TREE_COUNT"))).alias("QMD")
    ])
    TREEGRP = TREEGRP.with_columns([
        (np.square(pl.col("QMD")) * (0.005454 * (pl.col("TREE_COUNT")))).alias("BASAL AREA FROM QMD")
    ])

    TREEGRP = TREEGRP.with_columns([
        (pl.col("DIA_SQR_SUM") * 0.005454).alias("BASAL AREA FROM DBH")
    ])

    print(TREEGRP)

    #join out PLOT, SUBP, and COND tables together
    CONDGRP = PLOT.join(COND,on="PLT_CN", how="right")
    CONDGRP = CONDGRP.drop_nulls()
    CONDGRP = CONDGRP.unique("PLT_CN")
    print(f"CONDGRP with dropped nulls {CONDGRP}")


    #join our CONDGRP with our TREEGRP tables for final data frame
    FINAL = TREEGRP.join(CONDGRP, on="PLT_CN")
    FINAL = FINAL.with_columns([
        np.cos(np.radians(FINAL['ASPECT'])).alias("ASPECT_COS"),
        np.sin(np.radians(FINAL['ASPECT'])).alias("ASPECT_SIN"),
    ])
    FINAL = FINAL.with_columns(pl.col("PLT_CN").cast(pl.Int64))
    FINAL = FINAL.sort("PLT_CN")
    FINAL = FINAL.unique("PLT_CN")
    print(f"Final DataFrame {FINAL} ")

    print(FINAL.select(["PLT_CN","DIA_SQR_SUM", "QMD", "BALIVE","BASAL AREA FROM QMD", "BASAL AREA FROM DBH"]))
    FINAL.write_csv(os.path.join(data_dir, "output.csv"), separator=",") #write as csv for checks and records

    #test our filtering by giving it a real code and a fake code
    pondo = filter_by_forest_type(FINAL, 221)
    error = filter_by_forest_type(FINAL, 219)



    return FINAL
