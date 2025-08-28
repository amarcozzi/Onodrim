import torch
import polars as pl
from DataFrames import create_polars_dataframe_by_subplot

# Import shared modules
from utils import prepare_data, create_output_directories
from models import AttentionAutoencoder
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

WEIGHT_PATH = Path("./weights/BATCH1024attention_autoencoder.pt") #Path("./gini_coef_comparison/PLOTS_BOTH_GINI/BOTH_GINI_attention_autoencoder.pt") #Path("./weights/attention_autoencoder.pt")
INPUT_PATH = Path("./inference/input")
OUTPUT_PATH = Path("./inference/output")

CLIMATIC_PATH = Path("./data/climatic-interp-brd")
LANDFIRE_PATH = Path("./data/landfire")

CHUNK_SIZE = 64
BATCH_SIZE = 8

def get_feature_tile(feature_path, tile_height, tile_width, tile_transform, tile_crs):
    print(f"Sampling from {feature_path}")
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

def sanitize_layers(all_features, layer_indices, nodata_threshold=-1000, clip_min=-5, clip_max=5):
    for i in layer_indices:
        layer = all_features[:, :, i]

        # Replace nodata
        layer[layer < nodata_threshold] = 0.0

        # Replace NaNs or infs
        layer = np.nan_to_num(layer, nan=0.0, posinf=clip_max, neginf=clip_min)

        # Put back into all_features
        all_features[:, :, i] = layer
    return all_features

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

    print(f"Chunk size: {CHUNK_SIZE} Batch size: {BATCH_SIZE}")

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
    batch_size = 1024 #256
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

    # Find device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device) # Move to GPU
    model.eval()

    # Load scaler
    scaler = checkpoint['scaler']

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

        print(f"tile height: {tile_height} tile width: {tile_width}")

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

        # Convert basal area from square feet per pixel to square feet per acre
        tile_array[:, :, 0] = (tile_array[:, :, 0] / sq_m_per_pixel) * 4046.86

        # Convert max height from meters to feet
        tile_array[:, :, 1] = tile_array[:, :, 1] * 3.28084

        # Convert tree count from tree count per pixel to count per acre
        tile_array[:, :, 2] = (tile_array[:, :, 2] / sq_m_per_pixel) * 4046.86

        # Add features from UNET output
        all_features[:, :, 0] = tile_array[:, :, 2] # Tree count
        all_features[:, :, 1] = tile_array[:, :, 1] # Max height
        all_features[:, :, 2] = tile_array[:, :, 0] # Basal area
        
        all_features[:, :, 3:5] = tile_array[:, :, 3:5] # Gini dia and gini ht (they are in same order)

        # Add elev
        all_features[:, :, 5] = get_feature_tile(LANDFIRE_PATH / "LC20_Elev_220.tif", tile_height, tile_width, tile_transform, tile_crs)
        # all_features[:, :, 5] = get_feature_tile(LANDFIRE_PATH / "LC20_Elev_220.tif", tile_height, tile_width, tile_bounds, tile_crs)

        # Add slope
        all_features[:, :, 6] = get_feature_tile(LANDFIRE_PATH / "LC20_SlpP_220.tif", tile_height, tile_width, tile_transform, tile_crs)
        # all_features[:, :, 6] = get_feature_tile(LANDFIRE_PATH / "LC20_SlpP_220.tif", tile_height, tile_width, tile_bounds, tile_crs)

        
        # Add aspect
        aspect_raw = get_feature_tile(LANDFIRE_PATH / "LC20_Asp_220.tif", tile_height, tile_width, tile_transform, tile_crs)
        # aspect_raw = get_feature_tile(LANDFIRE_PATH / "LC20_Asp_220.tif", tile_height, tile_width, tile_bounds, tile_crs)

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

        all_features[:, :, 9] = lons
        all_features[:, :, 10] = lats

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

        # all_features[:, :, 17] = 0
        all_features = sanitize_layers(all_features, range(0, 30))

        print("all features shape: ", all_features.shape)

        # batch_patches = [] # Empty list for batches
        # batch_indices = [] # Empty list to store batch indices
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

                # Apply scaler
                patch_scaled = scaler.transform(patch_reshaped)
                print(patch_scaled)

                # # Add to batch list
                # batch_patches.append(patch_scaled)
                # batch_indices.append((y_start, x_start))

                # if len(batch_patches) == BATCH_SIZE:
                #     # Stack into tensor
                #     input_tensor = torch.from_numpy(np.stack(batch_patches).astype("float32"))

                #     # Move to device here if that needs to happen

                #     with torch.no_grad():
                #         latent_batch = model.encoder(input_tensor)

                #     # Add to output array
                #     for i, (y0, x0) in enumerate(batch_indices):
                #         latent_patch = latent_batch[i].cpu().numpy().reshape(CHUNK_SIZE, CHUNK_SIZE, latent_dim)
                #         output_array[y0:y0+CHUNK_SIZE, x0:x0+CHUNK_SIZE, :] = latent_patch
                    
                #     batch_patches = [] # Empty
                #     batch_indices = [] # Empty

                # Create tensor and move to device
                input_tensor = torch.from_numpy(patch_scaled.astype("float32"))#.unsqueeze(0)#.to(device)
                # print(input_tensor)

                with torch.no_grad():
                    latent = model.encoder(input_tensor)

                # Reshape and move back to cpu
                latent_patch = latent.cpu().numpy().reshape(CHUNK_SIZE, CHUNK_SIZE, latent_dim)
                # print(latent_patch)

                # Store in output array
                output_array[y_start:y_start+CHUNK_SIZE, x_start:x_start+CHUNK_SIZE, :] = latent_patch
        
        # # Run any remaining batches
        # if batch_patches:
        #     # Stack into tensor
        #     input_tensor = torch.from_numpy(np.stack(batch_patches).astype("float32"))

        #     # Move to device here if that needs to happen

        #     with torch.no_grad():
        #         latent_batch = model.encoder(input_tensor)

        #     # Add to output array
        #     for i, (y0, x0) in enumerate(batch_indices):
        #         latent_patch = latent_batch[i].cpu().numpy().reshape(CHUNK_SIZE, CHUNK_SIZE, latent_dim)
        #         output_array[y0:y0+CHUNK_SIZE, x0:x0+CHUNK_SIZE, :] = latent_patch
            
        #     batch_patches = [] # Empty
        #     batch_indices = [] # Empty

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