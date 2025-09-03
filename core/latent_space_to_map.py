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
import matplotlib.pyplot as plt

from models import AttentionAutoencoder, GatedAttention
from DataFrames import create_polars_dataframe_by_subplot
import database as db

AOI_PATH = Path("./inference/coconino")
STATE = "AZ"

LATENT_MAP_PATH = AOI_PATH / Path("./LATENT-output")
OUTPUT_PATH = AOI_PATH / Path("./ONODRIM-variable-maps")
WEIGHT_PATH = AOI_PATH / Path("./weights/sept2_gated_attention_autoencoder.pt")

UNET_TIF_PATH = AOI_PATH / Path("./UNET-input")

PLOT_ID_COL = 'SUBPLOTID'

KNN = 30 # k nearest neighbors

VAR_TO_MAP = "TREE_COUNT"

CATEGORICAL = False

# x_vals should be predicted, y_vals should be UNET.
def make_scatter(x_vals, y_vals, name, file_name):

    unet_diff = y_vals - x_vals

    min_val = min(x_vals.min(), y_vals.min())
    max_val = max(x_vals.max(), y_vals.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # regular scatter
    axes[0].scatter(x_vals, y_vals, s=5, alpha=0.5)
    axes[0].set_title(f"UNET vs {name} Predicted")
    axes[0].set_xlabel(f"{name} prediction")
    axes[0].set_ylabel("UNET prediction")
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1) # line
    axes[0].set_aspect("equal", adjustable="box")

    # residuals
    axes[1].scatter(x_vals, unet_diff, s=5, alpha=0.5, color="green")
    axes[1].set_title(f"Residuals (UNET - {name} prediction)")
    axes[1].set_xlabel(f"{name} prediction")
    axes[1].set_ylabel("Residuals")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1) # line at 0

    plt.tight_layout()
    file_name = f"./scatter/{name}_scatter_plots_{file_name}.png"
    plt.savefig(AOI_PATH / file_name, dpi=300, bbox_inches="tight")
    plt.close()

# Remove later
# For clipping. Max from FIA.
MAX_TPA = 1347 # Count per acre
MAX_MAX_HT = 191 # Feet
MAX_BASAL_AREA = 861 # feet^2 per acre
def clip_tile(layer, max):
    layer = np.nan_to_num(layer) # Replace nans with 0
    layer[layer < 0] = 0.0 # Clip to 0
    layer[layer > max] = np.nan # Replace values over a max with nans
    return layer

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

    # Load checkpoint first to configure model from saved metadata
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)

    # Feature columns used by this model (including bioclimatic variables)
    feature_cols = checkpoint.get('feature_cols')

    # Model and Training parameters (loaded from checkpoint when available)
    latent_dim = checkpoint.get('latent_dim')
    hidden_dims = checkpoint.get('hidden_dims')
    dropout_rate = checkpoint.get('dropout_rate')
    attention_module_name = checkpoint.get('attention_module')

    # Resolve attention module class if specified in checkpoint
    attention_modules = {
        'GatedAttention': GatedAttention
    }
    attention_module_cls = attention_modules.get(attention_module_name, None)

    # Load model
    input_dim = len(feature_cols)
    model = AttentionAutoencoder(
        input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        attention_module=attention_module_cls
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scaler
    scaler = checkpoint['scaler']
    # Make dataframe of all FIA plots
    plot_data = create_polars_dataframe_by_subplot(STATE, climate_resolution="10m")
    # plot_data.write_csv("./CSV_OF_PLOT_DATA_ALL.csv")
    # exit()

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

    # # Load in Old Growth CSV
    # OLD_GROWTH = pl.read_csv("./inference/coconino/subplot-ofe-AZ.csv")
    # # Create SUBPLOTID
    # OLD_GROWTH = OLD_GROWTH.with_columns(
    #                             pl.concat_str(
    #                             [
    #                                     pl.col("PLT_CN"),
    #                                     pl.col("SUBP")
    #                                     ],
    #                                     separator= "",
    #                                 ).alias("SUBPLOTID"),
    #                             )
    # # unique_subpids_ofe = len(OLD_GROWTH["SUBPLOTID"].unique())
    # # print(f"Unique: {unique_subpids_ofe} Len: {len(OLD_GROWTH['SUBPLOTID'])} Diff: {len(OLD_GROWTH['SUBPLOTID']) - unique_subpids_ofe}")
    # # exit()
    # # Cast to int64 for join
    # OLD_GROWTH = OLD_GROWTH.with_columns([
    #     pl.col(PLOT_ID_COL).cast(pl.Int64)
    # ])
    # print("old growth: ", OLD_GROWTH)

    # # print(OLD_GROWTH.select(PLOT_ID_COL).group_by(PLOT_ID_COL).agg(pl.len()).filter(pl.col("len") > 1))
    # # exit()

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
        file_name = f"{VAR_TO_MAP}_{latent.stem}.tif"
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
                crs = src.crs
                transform = src.transform
        except RasterioIOError as e:
            print(f"Error opening {file_name}. Skipping.")

        valid_mask = ~np.isnan(latent_array).any(axis=1) # Create mask of valid latent pixels
        distances = np.full((latent_array.shape[0], KNN), np.nan, dtype=float) # Create nan distances array 
        indices = np.full((latent_array.shape[0], KNN), -1, dtype=int) # Create indices array of all -1

        if np.any(valid_mask):
            distances_valid, indices_valid = fia_plot_tree.query(latent_array[valid_mask], k=KNN)

            # Add dimension for KNN==1
            if KNN == 1:
                distances_valid = distances_valid[:, np.newaxis]
                indices_valid = indices_valid[:, np.newaxis]
            
            # Put valid values back in distance and index arrays
            distances[valid_mask] = distances_valid
            indices[valid_mask] = indices_valid

        # Get nearest subplot ids for valid pixels
        nearest_spids = np.full(indices.shape, np.nan, dtype=int) # Make full of nans
        nearest_spids[valid_mask] = plot_ids_all[indices[valid_mask]]#.astype(str)
        flat_spids = nearest_spids.ravel()

        # Make into DF
        spid_df = pl.DataFrame({"SUBPLOTID": flat_spids})

        # OLD_GROWTH_unique = OLD_GROWTH.unique(subset=[PLOT_ID_COL]) # Taking out unique, temp fix!!!
        joined = spid_df.join(
            plot_data.select([PLOT_ID_COL, VAR_TO_MAP]), # This must be DF with plot id col and variable(s) we are mapping ***MUST CHANGE***
            on=PLOT_ID_COL,
            how="left"
        )
        print("joined:", joined)
        num_matches = joined.filter(pl.col(VAR_TO_MAP).is_not_null()).height
        total_rows = joined.height

        print(f"Matches: {num_matches}/{total_rows} ({num_matches/total_rows*100:.2f}%)")
        # exit()

        var_grid = joined[VAR_TO_MAP].to_numpy().reshape(nearest_spids.shape)

        # Waiting to deal with categorical for a sec
        # # Take mode of categorical
        # if CATEGORICAL:
        #     var_grid, counts = mode(var_grid, axis=1, nan_policy="omit")
        #     var_grid = var_grid.ravel()
        # # Else regular average
        # else:
        
        # FOR k=1
        if KNN > 1:
            var_avg = np.nanmean(var_grid, axis=-1).reshape(height, width)
            var_std = np.nanstd(var_grid, axis=-1).reshape(height, width)
        else:
            var_avg = var_grid.reshape(height, width)
            var_std = var_grid.reshape(height, width)

        # # Adding in UNET TPA diff band
        # full_unet_tif_path = UNET_TIF_PATH / f"{latent.stem}.tif"
        # print(f"UNET tif path: {full_unet_tif_path} Output file name: {output_file_path}")
        # tile_ds = rasterio.open(full_unet_tif_path)
        # tile_array = tile_ds.read()
        # tile_array = np.transpose(tile_array, (1, 2, 0)) # Transpose to (H, W, C)
        # sq_m_per_pixel = tile_ds.transform[0] **2

        # tile_array[:, :, 2] = (tile_array[:, :, 2] / sq_m_per_pixel) * 4046.86 # Convert tree count from tree count per pixel to count per acre
        # tile_array[:, :, 0] *= 10.7639  # Convert basal area from square meters to square feet
        # tile_array[:, :, 0] = (tile_array[:, :, 0] / sq_m_per_pixel) * 4046.86  # sq ft per acre
        # tile_array[:, :, 1] = tile_array[:, :, 1] * 3.28084

        # # Clip to 0 and replace outliers with nans
        # tile_array[:, :, 0] = clip_tile(tile_array[:, :, 0], MAX_BASAL_AREA)
        # tile_array[:, :, 1] = clip_tile(tile_array[:, :, 1], MAX_MAX_HT)
        # tile_array[:, :, 2] = clip_tile(tile_array[:, :, 2], MAX_TPA)

        # if VAR_TO_MAP == "TREE_COUNT":
        #     unet_var = tile_array[:, :, 2]
        # elif VAR_TO_MAP == "BASAL_AREA_TREE":
        #     unet_var = tile_array[:, :, 0]
        # elif VAR_TO_MAP == "MAX_HT":
        #     unet_var = tile_array[:, :, 1]
        # else:
        #     print("Unaccounted for variable to map. Please fix. Exiting.")
        #     exit()
        # # unet_var = unet_var * 0.5

        # # Create UNET - TPA PREDICTED band
        # unet_diff = unet_var - var_avg

        # x_vals = var_avg.flatten()
        # y_vals = unet_var.flatten()

        # not_nan_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        # make_scatter(x_vals[not_nan_mask], y_vals[not_nan_mask], f"ONODRIM_{VAR_TO_MAP}_K{KNN}", latent.stem)
        # exit() # EXIT AFTER SCATTER
        print(var_avg)
        band_stack = np.stack((var_avg.astype(np.float32), var_std.astype(np.float32)), axis=-1)

        band_stack = np.transpose(band_stack, (2, 0, 1))

        print(f"Var to map: {VAR_TO_MAP} KNN: {KNN} File: {output_file_path}")

        with rasterio.open(
            output_file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=band_stack.shape[0],
            dtype=np.float32,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(band_stack)
        exit()


if __name__ == "__main__":
    main()
