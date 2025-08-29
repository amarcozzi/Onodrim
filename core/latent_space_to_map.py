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


LATENT_MAP_PATH = Path("./inference/output")
OUTPUT_PATH = Path("./inference/variable-maps")
WEIGHT_PATH = Path("./weights/NOISYgated_attention_autoencoder.pt")

PLOT_ID_COL = 'SUBPLOTID'

KNN = 10 # k nearest neighbors

VAR_TO_MAP = "TREE_COUNT" #'FORTYPCD'

CATEGORICAL = False

STATE = "MT"

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
    plt.savefig(f"./inference/{name}_scatter_plots_{file_name}.png", dpi=300, bbox_inches="tight")
    plt.close()

# Remove later
def clip_negs(all_features, layer_indices):
    for i in layer_indices:
        layer = all_features[:, :, i]
        layer = np.nan_to_num(layer) # Replace nans with 0
        layer[layer < 0] = 0.0 # Clip to 0
        all_features[:, :, i] = layer
    return all_features

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
        file_name = f"ONODRIM_VAR_MATCH{latent.stem}.tif"
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
        print(f"var grid shape: {var_grid.shape}")

        # Waiting to deal with categorical for a sec
        # # Take mode of categorical
        # if CATEGORICAL:
        #     var_grid, counts = mode(var_grid, axis=1, nan_policy="omit")
        #     var_grid = var_grid.ravel()
        # # Else regular average
        # else:
        
        # FOR k=1
        if KNN > 1:
            var_avg = np.mean(var_grid, axis=-1).reshape(height, width)
            var_std = np.std(var_grid, axis=-1).reshape(height, width)
        else:
            var_avg = var_grid.reshape(height, width)
            var_std = var_grid.reshape(height, width)

        print(f"var avg shape: {var_avg.shape}")

        # Adding in UNET TPA diff band
        unet_tif_path = Path("./inference/input") / f"{latent.stem}.tif"
        print(f"UNET tif path: {unet_tif_path} Output file name: {output_file_path}")
        tile_ds = rasterio.open(unet_tif_path)
        tile_array = tile_ds.read()
        tile_array = np.transpose(tile_array, (1, 2, 0)) # Transpose to (H, W, C)
        sq_m_per_pixel = tile_ds.transform[0] **2
        # tile_array[:, :, 2] = (tile_array[:, :, 2] / sq_m_per_pixel) * 4046.86 # Convert tree count from tree count per pixel to count per acre
        tile_array[:, :, 0] *= 10.7639  # Convert basal area from square meters to square feet
        tile_array[:, :, 0] = (tile_array[:, :, 0] / sq_m_per_pixel) * 4046.86  # sq ft per acre
        tile_array = clip_negs(tile_array, [0]) # Clip negatives from tree count band
        unet_var = tile_array[:, :, 0]

        # Create UNET - TPA PREDICTED band
        unet_diff = unet_var - var_avg

        x_vals = var_avg.flatten()
        y_vals = unet_var.flatten()

        make_scatter(x_vals, y_vals, f"ONODRIM_BASAL_K{KNN}", latent.stem)
        exit() # EXIT AFTER SCATTER

        band_stack = np.stack((var_avg, var_std, unet_diff), axis=-1)

        print(f"MEAN OF ABS VALUE: {np.mean(np.abs(band_stack[:, :, 2]))}")

        print(f"Band stack shape: {band_stack.shape}")
        band_stack = np.transpose(band_stack, (2, 0, 1))

        print(f"Band stack shape: {band_stack.shape}")

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
