import numpy as np
import torch
from pathlib import Path
import rasterio
from tqdm import tqdm
from scipy.spatial import cKDTree
import sqlite3
import polars as pl
from rasterio.errors import RasterioIOError
from scipy.stats import mode

from models import AttentionAutoencoder
from DataFrames import create_polars_dataframe_by_subplot
import database as db


LATENT_MAP_PATH = Path("./inference/output")
OUTPUT_PATH = Path("./inference/variable-maps")
WEIGHT_PATH = Path("./gini_coef_comparison/PLOTS_BOTH_GINI/BOTH_GINI_attention_autoencoder.pt")

PLOT_ID_COL = 'SUBPLOTID'

KNN = 4 # k nearest neighbors

VAR_TO_MAP = "TREE_COUNT" #'FORTYPCD'

WEIGHTED = True

STATE = "MT"

def main():
 
    # Check paths
    if not LATENT_MAP_PATH.exists():
        print(f"Input path {LATENT_MAP_PATH} does not exist, exiting.")
        exit()
    if not OUTPUT_PATH.exists():
        print(f"Output path {OUTPUT_PATH} does not exist, exiting.")
        exit()
    if not WEIGHT_PATH.exists():
        print(f"Weight path {WEIGHT_PATH} does not exist, exiting.")
        exit()

    # Feature columns used by this model (including bioclimatic variables)
    feature_cols = [
        "TREE_COUNT",
        'MAX_HT',
        'BASAL_AREA_TREE',
        'GINI_DIA',
        'GINI_HT',
        'ELEV',
        'SLOPE',
        'ASPECT_COS',
        'ASPECT_SIN',
        "LAT",
        "LON",
        "MEAN_TEMP",  # BIO1   ANNUAL MEAN TEMP
        "MEAN_DIURNAL_RANGE",  # BIO2   MEAN OF MONTHLY (MAX TEMP _ MIN TEMP)
        "ISOTHERMALITY",  # BIO3   (BIO2/BIO7)*100
        "TEMP_SEASONALITY",  # BIO4   (STD DEV * 100)
        "MAX_TEMP_WARM_MONTH",  # BIO5
        "MIN_TEMP_COLD_MONTH",  # BIO6
        "TEMP_RANGE",  # BIO7   (BIO5 - BIO6)
        "MEAN_TEMP_WET_QUARTER",  # BIO8
        "MEAN_TEMP_DRY_QUARTER",  # BIO9
        "MEAN_TEMP_WARM_QUARTER",  # BIO10
        "MEAN_TEMP_COLD_QUARTER",  # BIO11
        "ANNUAL_PRECIP",  # BIO12
        "PRECIP_WET_MONTH",  # BIO13
        "PRECIP_DRY_MONTH",  # BIO14
        "PRECIP_SEASONALITY",  # BIO15  (COEFFICIENT of VARIATION)
        "PRECIP_WET_QUARTER",  # BIO16
        "PRECIP_DRY_QUARTER",  # BIO17
        "PRECIP_WARM_QUARTER",  # BIO18
        "PRECIP_COLD_QUARTER"  # BIO19
    ]

    # Model and Training parameters
    latent_dim = 16
    hidden_dims = [64]
    batch_size = 256
    learning_rate = 1e-4
    num_epochs = 10000
    dropout_rate = 0.2
    use_attention = True

    # Load model
    input_dim = len(feature_cols)
    model = AttentionAutoencoder(
        input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
        dropout_rate=dropout_rate, use_attention=use_attention
    )

    # Load checkpoint
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device) # Move to GPU
    model.eval()

    # Load scaler
    scaler = checkpoint['scaler']

    # Make dataframe of all FIA plots
    plot_data = create_polars_dataframe_by_subplot("MT", climate_resolution="10m")

    # Get encoded version of this
    ALL_PLOT = plot_data.select(feature_cols).to_numpy()
    ALL_PLOT = np.nan_to_num(ALL_PLOT, nan=0.0)
    all_plot_scaled = scaler.transform(ALL_PLOT)
    plot_ids_all = plot_data.select(PLOT_ID_COL).to_numpy().flatten()
    all_plot_tensor = torch.tensor(all_plot_scaled, dtype=torch.float32)

    with torch.no_grad():
        all_embeddings = model.encoder(all_plot_tensor).numpy()

    # Make KD tree of plots
    fia_plot_tree = cKDTree(all_embeddings)

    # If we are not using stuff that has cols in plot_data, must make plot
    # In future make giant plot_data with everything
    # CREATE DF THAT CONTAINS AT LEAST SUBPLOTID AND VARS_TO_MAP HERE ###################################
    # # Make dataframe that is SUBPLOTID and FORTYPCD
    # # * Sub COND for any table in the future?
    # COND = db.get_df_from_db(STATE, "COND", [VAR_TO_MAP, "PLT_CN"])
    # COND = COND.sort("PLT_CN")

    # SUBP = db.get_df_from_db(STATE, "SUBPLOT", ["PLT_CN", "SUBP"])
    # SUBP = SUBP.join(COND, on="PLT_CN", how="right", coalesce=True)
    # SUBP = SUBP.drop_nulls()
    # SUBP = SUBP.with_columns(
    #                             pl.concat_str(
    #                             [
    #                                     pl.col("PLT_CN"),
    #                                     pl.col("SUBP")
    #                                     ],
    #                                     separator= "",
    #                                 ).alias("SUBPLOTID"),
    #                             )
    # SUBP = SUBP.sort("SUBPLOTID")
    # print(SUBP)

    # #  subp_map = dict(zip(SUBP["SUBPLOTID"], SUBP[VAR_TO_MAP]))
    ######################################################################################################

    latents = list(LATENT_MAP_PATH.glob("*.tif"))
    for latent in tqdm(latents):
        # Get output file name
        file_name = f"{latent.stem}.tif"
        output_file_path = OUTPUT_PATH / file_name

        # Open latent space raster
        try:
            with rasterio.open(latent) as src:
                latent_array = src.read() # (bands, height, width)
                latent_array = np.transpose(latent_array, (1, 2, 0))  # (height, width, bands)
                latent_array = latent_array.reshape(-1, src.count)
                profile = src.profile
                height, width = src.height, src.width
                latent_dtype = src.dtypes[0]
        except RasterioIOError as e:
            print(f"Error opening {file_name}. Skipping.")

        # * Change this to do a weighted average based on distance
        distances, indices = fia_plot_tree.query(latent_array, k=KNN)

        # Get nearest subplot ids
        nearest_spids = plot_ids_all[indices]#.astype(str)
        flat_spids = nearest_spids.ravel()

        # Make into DF
        spid_df = pl.DataFrame({"SUBPLOTID": flat_spids})

        joined = spid_df.join(
            plot_data.select([PLOT_ID_COL, VAR_TO_MAP]), # This must be DF with plot id col and variable(s) we are mapping
            on=PLOT_ID_COL,
            how="left"
        )

        var_grid = joined[VAR_TO_MAP].to_numpy().reshape(nearest_spids.shape)

        # neighbor_values = np.array([[subp_map.get(sid, np.nan) for sid in row] for row in nearest_spids])

        # Weighted average
        if WEIGHTED:
            weights = 1 / (distances + 1e-6)
            var_grid = np.nansum(var_grid * weights, axis=1) / np.nansum(weights, axis=1)
        # Mode for categorical
        else:
            var_grid, counts = mode(var_grid, axis=1, nan_policy="omit")
            var_grid = var_grid.ravel()

        # Reshape to tile shape
        var_grid = var_grid.reshape(height, width)


        # Update tif profile
        profile.update(
            dtype=latent_dtype,
            count=1,
            compress="lzw"
        )

        # Write out
        with rasterio.open(output_file_path, "w", **profile) as dst:
            dst.write(var_grid.astype(latent_dtype), 1)
        exit()


if __name__ == "__main__":
    main()
