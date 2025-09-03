import torch
import polars as pl
from DataFrames import create_polars_dataframe_by_subplot

# Import shared modules
from dataloader import prepare_data, create_output_directories
from models import AttentionAutoencoder, GatedAttention
from train import train_autoencoder
from evaluation import run_evaluation

from pathlib import Path
from tqdm import tqdm
import rasterio
import numpy as np
from skimage.transform import resize
from pyproj import Transformer
from sklearn.preprocessing import StandardScaler
from rasterio.errors import RasterioIOError
import rioxarray
from rasterio.transform import xy
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.interpolate import interpn
import xarray

from scipy.spatial import cKDTree


import matplotlib.pyplot as plt

AOI_PATH = Path("./inference/coconino")
STATE = "AZ"
WEIGHT_PATH = AOI_PATH / Path("./weights/sept2_gated_attention_autoencoder.pt") #Path("./gini_coef_comparison/PLOTS_BOTH_GINI/BOTH_GINI_attention_autoencoder.pt") #Path("./weights/attention_autoencoder.pt")
INPUT_PATH = AOI_PATH / Path("./UNET-input")
OUTPUT_PATH = AOI_PATH / Path("./LATENT-output")

CLIMATIC_PATH = AOI_PATH / Path("./climatic-interpolated")
LANDFIRE_PATH = Path("data/landfire")

CHUNK_SIZE = 64

# For clipping. Max from FIA.
MAX_TPA = 1347 # Count per acre
MAX_MAX_HT = 191 # Feet
MAX_BASAL_AREA = 861 # feet^2 per acre

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

def get_feature_tile(feature_path, tile_height, tile_width, tile_transform, tile_crs):
    # print(f"Sampling from {feature_path}")
    feature = rioxarray.open_rasterio(feature_path)

    rows, cols = np.meshgrid(np.arange(tile_height), np.arange(tile_width), indexing='ij')
    xs, ys = rasterio.transform.xy(tile_transform, rows, cols, offset='center')
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    transformer = Transformer.from_crs(tile_crs, feature.rio.crs, always_xy=True)
    xx_flat, yy_flat = transformer.transform(xs, ys)

    xx = xx_flat.reshape(tile_height, tile_width)
    yy = yy_flat.reshape(tile_height, tile_width)

    xx_da = xarray.DataArray(xx, dims=("y","x"))
    yy_da = xarray.DataArray(yy, dims=("y","x"))

    sampled = feature.interp(x=xx_da, y=yy_da, method="nearest")

    sampled_array = sampled.values.reshape(tile_height, tile_width)

    return sampled_array

def clip_tile(layer, max):
    layer = np.nan_to_num(layer) # Replace nans with 0
    layer[layer <= 0] = np.nan # Replace negs and 0s with nans
    layer[layer > max] = np.nan # Replace values over a max with nans
    return layer

def main():
    # Check paths
    if not INPUT_PATH.exists():
        print(f"Input path {INPUT_PATH} does not exist, exiting.")
        exit()
    if not OUTPUT_PATH.exists():
        print(f"Output path {OUTPUT_PATH} does not exist, exiting.")
        exit()
    if not WEIGHT_PATH.exists():
        print(f"Weight path {WEIGHT_PATH} does not exist, exiting.")
        exit()

    print(f"Chunk size: {CHUNK_SIZE}")

    # Load checkpoint first to configure model from saved metadata
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)

    # Feature columns used by this model (including bioclimatic variables)
    feature_cols = checkpoint.get('feature_cols')

    # Model and Training parameters (loaded from checkpoint when available)
    latent_dim = checkpoint.get('latent_dim')
    hidden_dims = checkpoint.get('hidden_dims')
    dropout_rate = checkpoint.get('dropout_rate')
    attention_module_name = checkpoint.get('attention_module')
    scaler = checkpoint['scaler']

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

    # Loop through all tiles in input directory
    tiles = list(INPUT_PATH.glob("*.tif"))
    for tile in tqdm(tiles):
        file_name = f"{tile.stem}.tif"
        output_file_path = OUTPUT_PATH / file_name

        # Get info about tile
        tile_ds = rasterio.open(tile)
        tile_crs = tile_ds.crs
        tile_transform = tile_ds.transform
        tile_height = tile_ds.height
        tile_width = tile_ds.width
        sq_m_per_pixel = tile_ds.transform[0] **2

        # print(f"tile height: {tile_height} tile width: {tile_width}")

        # Read in entire tile
        try:
            tile_array = tile_ds.read()
        except RasterioIOError as e:
            print(f"Error reading {tile.stem}. Skipping.")
            continue

        tile_array = np.transpose(tile_array, (1, 2, 0)) # Transpose to (H, W, C)

        tile_bounds = rasterio.windows.bounds(
            rasterio.windows.Window(0, 0, tile_width, tile_height),
            tile_transform
        )

        # Create output array the same shape as the input but with latent_dim bands
        output_array = np.zeros([tile_height, tile_width, latent_dim])

        # Preallocate array for arrays
        all_features = np.zeros((tile_height, tile_width, len(feature_cols)), dtype=np.float32)

        # Convert basal area from square meters per pixel to square feet per acre
        tile_array[:, :, 0] *= 10.7639  # Convert basal area from square meters to square feet
        tile_array[:, :, 0] = (tile_array[:, :, 0] / sq_m_per_pixel) * 4046.86  # sq ft per acre

        # Convert max height from meters to feet
        tile_array[:, :, 1] = tile_array[:, :, 1] * 3.28084

        # Convert tree count from tree count per pixel to count per acre
        tile_array[:, :, 2] = (tile_array[:, :, 2] / sq_m_per_pixel) * 4046.86

        # Clip to 0 and replace outliers with nans
        tile_array[:, :, 0] = clip_tile(tile_array[:, :, 0], MAX_BASAL_AREA)
        tile_array[:, :, 1] = clip_tile(tile_array[:, :, 1], MAX_MAX_HT)
        tile_array[:, :, 2] = clip_tile(tile_array[:, :, 2], MAX_TPA)

        # Add features from UNET output
        all_features[:, :, 0] = tile_array[:, :, 2] # Tree count
        all_features[:, :, 1] = tile_array[:, :, 1] # Max height
        all_features[:, :, 2] = tile_array[:, :, 0] # Basal area
        
        all_features[:, :, 3:5] = tile_array[:, :, 3:5] # Gini dia and gini ht (they are in same order)

        # Add elev
        all_features[:, :, 5] = get_feature_tile(LANDFIRE_PATH / "LC20_Elev_220.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 5] = all_features[:, :, 5] * 3.28084  # Convert elev from meters to feet

        # Add slope
        all_features[:, :, 6] = get_feature_tile(LANDFIRE_PATH / "LC20_SlpP_220.tif", tile_height, tile_width, tile_transform, tile_crs)
 
        # Add aspect
        aspect_raw = get_feature_tile(LANDFIRE_PATH / "LC20_Asp_220.tif", tile_height, tile_width, tile_transform, tile_crs)

        aspect_rad = np.deg2rad(aspect_raw)
        aspect_cos = np.cos(aspect_rad)
        aspect_sin = np.sin(aspect_rad)
        all_features[:, :, 7] = aspect_cos
        all_features[:, :, 8] = aspect_sin

        # Add lat and lon
        # Get lat lon arrays and add to feature array
        rows, cols = np.meshgrid(np.arange(tile_height), np.arange(tile_width), indexing='ij')

        # Compute coordinates in tile CRS
        xs, ys = rasterio.transform.xy(tile_transform, rows, cols, offset='center')

        # Reshape to be the same size as tile
        xs = np.array(xs, dtype=np.float32).reshape(tile_height, tile_width)
        ys = np.array(ys, dtype=np.float32).reshape(tile_height, tile_width)

        # Transform to EPSG:4326 (lat lon)
        transformer = Transformer.from_crs(tile_ds.crs, "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)

        all_features[:, :, 9] = lats
        all_features[:, :, 10] = lons

        # Add all climatic variables
        all_features[:, :, 11] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_1.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 12] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_2.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 13] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_3.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 14] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_4.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 15] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_5.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 16] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_6.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 17] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_7.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 18] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_8.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 19] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_9.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 20] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_10.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 21] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_11.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 22] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_12.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 23] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_13.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 24] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_14.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 25] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_15.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 26] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_16.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 27] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_17.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 28] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_18.tif", tile_height, tile_width, tile_transform, tile_crs)
        all_features[:, :, 29] = get_feature_tile(CLIMATIC_PATH / "wc2.1_10m_bio_19.tif", tile_height, tile_width, tile_transform, tile_crs)


        ######################################################
        # # SKLEARN SCALAR FIT
        # VAR_TO_PLOT = "OFE"
        # plot_data = create_polars_dataframe_by_subplot(STATE, climate_resolution="10m")
        # fia_plot_ids = plot_data.select("SUBPLOTID").to_numpy().flatten() # Save off plot ids
        # plot_data_features = plot_data[feature_cols] # Select feature cols in correct order
        # # # Print FIA maxes
        # # print(f"Max TPA: {plot_data_features['TREE_COUNT'].max()} "
        # #         f"Max height: {plot_data_features['MAX_HT'].max()} "
        # #         f"Max basal area: {plot_data_features['BASAL_AREA_TREE'].max()}")
        # # exit()
        # plot_data_features.write_csv("./CSV_OF_PLOT_DATA_FEATURES.csv")
        # # exit()

        # plot_data_array = plot_data_features.to_numpy() # Make into numpy array

        # all_features_reshaped = all_features.reshape(-1, all_features.shape[-1]) # Reshape to (w*h, features)
        # print(f"All features shape: {all_features_reshaped.shape} Plot data array shape: {plot_data_array.shape}")

        # # Scale both
        # sc = StandardScaler()
        # fia_data_scaled = sc.fit_transform(plot_data_array)
        # tile_data_scaled = sc.transform(all_features_reshaped)

        # # Make KD tree from FIA
        # fia_plot_tree = cKDTree(fia_data_scaled)

        # KNN = 1
        # valid_mask = ~np.isnan(tile_data_scaled).any(axis=1) # Create mask of valid latent pixels
        # distances = np.full((tile_data_scaled.shape[0], KNN), np.nan, dtype=float) # Create nan distances array 
        # indices = np.full((tile_data_scaled.shape[0], KNN), -1, dtype=int) # Create indices array of all -1

        # # Get indices of closest
        # if np.any(valid_mask):
        #     distances_valid, indices_valid = fia_plot_tree.query(tile_data_scaled[valid_mask], k=KNN)

        #     # Add dimension for KNN==1
        #     if KNN == 1:
        #         distances_valid = distances_valid[:, np.newaxis]
        #         indices_valid = indices_valid[:, np.newaxis]
            
        #     # Put valid values back in distance and index arrays
        #     distances[valid_mask] = distances_valid
        #     indices[valid_mask] = indices_valid

        # # Get nearest subp ids by backtracking indices
        # nearest_subp_ids = fia_plot_ids[indices]

        # flat_spids = nearest_subp_ids.ravel()

        # # Make into DF
        # spid_df = pl.DataFrame({"SUBPLOTID": flat_spids})

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
        #     pl.col("SUBPLOTID").cast(pl.Int64)
        # ])
        # print("old growth: ", OLD_GROWTH)
        # OLD_GROWTH_unique = OLD_GROWTH.unique(subset=["SUBPLOTID"]) # Taking out unique, temp fix!!!

        # joined = spid_df.join(
        #     OLD_GROWTH_unique.select(["SUBPLOTID", VAR_TO_PLOT]), # This must be DF with plot id col and variable(s) we are mapping
        #     on="SUBPLOTID",
        #     how="left"
        # )

        # var_grid = joined[VAR_TO_PLOT].to_numpy().reshape(nearest_subp_ids.shape) # Isolate var
        # print(f"var grid shape: {var_grid.shape}")

        # var_avg = np.nanmean(var_grid, axis=-1).reshape(tile_height, tile_width) # Calculate mean
        # var_std = np.nanstd(var_grid, axis=-1).reshape(tile_height, tile_width) # Calculate std dev
        # print(f"var avg shape: {var_avg.shape}")

        # # Create UNET - TPA PREDICTED band
        # unet_diff = all_features[:, :, 0] - var_avg

        # band_stack = np.stack((var_avg, var_std, unet_diff), axis=-1) # Stack into bands
        # print(f"Band stack shape: {band_stack.shape}")

        # print(f"MEAN OF ABS VALUE: {np.mean(np.abs(band_stack[:, :, 2]))}")

        # band_stack = np.transpose(band_stack, (2, 0, 1)) # Put band dimension first for write out
        # print(f"Band stack shape: {band_stack.shape}")

        # standard_scalar_test_path_name = OUTPUT_PATH / f"{VAR_TO_PLOT}STANDARD_SCALER_TEST{file_name}.tif"

        ########################################################
        # # MAKE SCATTER PLOTS
        # COMP_NAME = "SCALER"
        # x_vals = var_avg.flatten()
        # y_vals = all_features[:, :, 0].flatten()

        # make_scatter(x_vals, y_vals, COMP_NAME)

        #####################################################
        # # Write scaler geotif (AFTER SCALAR STUFF)
        # with rasterio.open(
        #     standard_scalar_test_path_name,
        #     'w',
        #     driver='GTiff',
        #     height=tile_height,
        #     width=tile_width,
        #     count=band_stack.shape[0],
        #     dtype=np.float32,
        #     crs=tile_crs,
        #     transform=tile_transform
        # ) as dst:
        #     dst.write(band_stack)
        # exit()

        ######################################################
        # # HISTOGRAMS
        # # # All feature values vs FIA
        # # plot_data = create_polars_dataframe_by_subplot("MT", climate_resolution="10m")
        # # plot_data_features = plot_data[feature_cols]
        # # num_bins = 100
        # # print(f"FOR TILE {file_name}")
        # # for i, feature in enumerate(plot_data_features.columns):
        # #     print(f"Making plot for {feature}")
        # #     values_fia = plot_data_features[feature].to_numpy()
        # #     values_tile = all_features[:, :, i].flatten()
        # #     plt.figure(figsize=(6, 4))
        # #     xmin = min(values_fia.min(), values_tile.min())
        # #     xmax = max(values_fia.max(), values_tile.max())
        # #     plt.hist(values_fia, bins=num_bins, histtype="step", density=False, color="green", label="FIA plots", range=(xmin, xmax))
        # #     plt.hist(values_tile, bins=num_bins, histtype="step", density=False, color="purple", label="Tile values", range=(xmin, xmax))
        # #     plt.title(f"Hist for FIA plt {feature}")
        # #     plt.xlabel(feature)
        # #     plt.ylabel("Frequency")
        # #     # plt.show()
        # #     plt.savefig(f"./inference/histograms/hist_{feature}_{file_name}.png", dpi=300, bbox_inches="tight")
        # #     plt.close()
        # # exit()

        # num_bins = 100
        # print(f"FOR TILE {file_name}")
        # for i in range(0, 3):
        #     print(f"Making plot for {i}")
        #     values_tile = all_features[:, :, i].flatten()
        #     plt.figure(figsize=(6, 4))
        #     xmin = values_tile.min()
        #     xmax = values_tile.max()
        #     plt.hist(values_tile, bins=num_bins, histtype="step", density=False, color="purple", label="Tile values", range=(xmin, xmax))
        #     plt.title(f"Hist for FIA plt {i}")
        #     plt.xlabel(i)
        #     plt.ylabel("Frequency")
        #     # plt.show()
        #     plt.savefig(f"./inference/hist_JUST_UNET_{i}_{file_name}.png", dpi=300, bbox_inches="tight")
        #     plt.close()
        # exit()
        ######################################################

        for y in range(0, tile_height, CHUNK_SIZE):
            for x in range(0, tile_width, CHUNK_SIZE):
                # If we are on the edge, make starts be CHUNK_SIZE from edge
                y_start = y
                x_start = x
                if tile_height - y_start < CHUNK_SIZE:
                    y_start = tile_height - CHUNK_SIZE
                if tile_width - x_start < CHUNK_SIZE:
                    x_start = tile_width - CHUNK_SIZE

                # Get patch
                patch = all_features[y_start:y_start+CHUNK_SIZE, x_start:x_start+CHUNK_SIZE, :]

                # Reshape
                patch_reshaped = patch.reshape(-1, patch.shape[-1])

                valid_mask = ~np.isnan(patch_reshaped).any(axis=1) # Mask where entire vector is valid

                latent_patch = np.full((patch_reshaped.shape[0], latent_dim), np.nan, dtype=float) # Create full nan "latent" patch to fill in later

                if np.any(valid_mask):

                    # Apply scaler
                    patch_scaled = scaler.transform(patch_reshaped[valid_mask])

                    # Create tensor and move to device
                    input_tensor = torch.from_numpy(patch_scaled.astype("float32"))#.unsqueeze(0)#.to(device)

                    with torch.no_grad():
                        latent_valid = model.encoder(input_tensor)

                    # Reshape and move back to cpu
                    latent_patch[valid_mask] = latent_valid # Fill in latent patch with valid pixels

                latent_patch = latent_patch.reshape(CHUNK_SIZE, CHUNK_SIZE, latent_dim) # Reshape

                # Store in output array
                output_array[y_start:y_start+CHUNK_SIZE, x_start:x_start+CHUNK_SIZE, :] = latent_patch

        # Transpose back
        output_array = np.transpose(output_array, (2, 0, 1))

        # Write output array to geotif
        with rasterio.open(
            output_file_path,
            'w',
            driver='GTiff',
            height=tile_height,
            width=tile_width,
            count=latent_dim,
            dtype=np.float32,
            crs=tile_crs,
            transform=tile_transform
        ) as dst:
            dst.write(output_array)
        # exit()

if __name__ == "__main__":
    main()