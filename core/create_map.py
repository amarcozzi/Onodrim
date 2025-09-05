from pathlib import Path
import rasterio
from rasterio.merge import merge
import rasterio.mask
import geopandas as gpd
import numpy as np
from tqdm import tqdm

AOI = "coconino"
AOI_PATH = Path(f"./inference/{AOI}")
TILE_PATH = AOI_PATH / Path("./SCALER-variable-maps/QMD_TREE")
MOSAIC_PATH = AOI_PATH / Path("./mosaics")
SHAPE_FILE_PATH = AOI_PATH / Path("coconino_shape.geojson")

MOSAIC_FILE_NAME = "QMD_TREE_coconino_k10_weightedmean.tif"
TEMP_FILE_NAME = "temp.tif"

NODATA_VALUE = -9999

def mean_merge(merged_data, new_data, merged_mask, new_mask, index=None, roff=None, coff=None):
    # Mask out invalid pixels and nans
    valid_merged = ~merged_mask & (~np.isnan(merged_data)) & (merged_data > 0)
    valid_new = ~new_mask & (~np.isnan(new_data)) & (new_data > 0)

    # Keep mean where both data are valid
    both_valid = valid_merged & valid_new
    merged_data[both_valid] = (merged_data[both_valid] + new_data[both_valid]) / 2.0

    # Keep new and merged data when either is the only valid data
    only_new_valid = ~valid_merged & valid_new
    merged_data[only_new_valid] = new_data[only_new_valid]

    only_merged_valid = ~valid_new & valid_merged
    merged_data[only_merged_valid] = merged_data[only_merged_valid]

    return merged_data

def main():
    # Check paths 
    if not TILE_PATH.exists():
        print(f"Input path {TILE_PATH} does not exist, exiting.")
        exit()
    if not MOSAIC_PATH.exists():
        print(f"Output path {MOSAIC_PATH} does not exist, exiting.")
        exit()
    if not SHAPE_FILE_PATH.exists():
        print(f"Weight path {SHAPE_FILE_PATH} does not exist, exiting.")
        exit()

    print(f"Creating mosaic for {AOI} from {TILE_PATH}.")

    # Load shapefile
    shapefile_gdf = gpd.read_file(SHAPE_FILE_PATH)

    # Get all tif files
    tiles = list(TILE_PATH.glob("*.tif"))
    tiles_src = [rasterio.open(tile) for tile in tiles]

    # Merge all tiles together
    mosaic, out_transform = merge(tiles_src, nodata=NODATA_VALUE, method=mean_merge)
    mosaic = np.where(np.isnan(mosaic), NODATA_VALUE, mosaic)

    # Prepare meta data
    out_meta = tiles_src[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0],
        "nodata": NODATA_VALUE
    })

    # Write temp mosiac to tif
    temp_file_path = MOSAIC_PATH / TEMP_FILE_NAME
    with rasterio.open(temp_file_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    # Read in mosaic as Dataset Reader object
    with rasterio.open(temp_file_path) as mosaic_ds:
        shapefile_gdf = shapefile_gdf.to_crs(mosaic_ds.crs) # Get shapefile into correct CRS
        shapes = [feature["geometry"] for feature in shapefile_gdf.__geo_interface__["features"]]

        # https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html See link for parameter definitions
        cropped, cropped_transform = rasterio.mask.mask(mosaic_ds, shapes, all_touched=True, nodata=NODATA_VALUE, filled=True, crop=True)

    # Replace nans within the AOI with -9999
    # cropped = np.where(np.isnan(cropped), -9999, cropped)

    # Update metadata again
    out_meta.update({
        "driver": "GTiff",
        "height": cropped.shape[1],
        "width": cropped.shape[2],
        "transform": cropped_transform,
        "count": cropped.shape[0],
        "nodata": NODATA_VALUE
    })

    # Write final cropped mosaic to tif
    output_file_path = MOSAIC_PATH / MOSAIC_FILE_NAME
    with rasterio.open(output_file_path, "w", **out_meta) as dst:
        dst.write(cropped)

    for src in tiles_src:
        src.close()

if __name__ == "__main__":
    main()
# def clip_tile(layer, max):
#     layer = np.nan_to_num(layer) # Replace nans with 0
#     layer[layer <= 0] = np.nan # Replace negs and 0s with nans
#     layer[layer > max] = np.nan # Replace values over a max with nans
#     return layer
# CONVERTING WHOLE UNET TIF INTO FIA UNITS
# def clip_tile(layer, max):
#     layer = np.nan_to_num(layer) # Replace nans with 0
#     layer[layer <= 0] = np.nan # Replace negs and 0s with nans
#     layer[layer > max] = np.nan # Replace values over a max with nans
#     return layer

# 


    # exit()


     # MAX_TPA = 1347 # Count per acre
    # MAX_MAX_HT = 191 # Feet
    # MAX_BASAL_AREA = 861 # feet^2 per acre

    # output_file_path = MOSAIC_PATH / f"UNET_FIA_UNITS_PROP_NANS.tif"

    # map_array_ds = rasterio.open(MOSAIC_PATH / MOSAIC_FILE_NAME)
    # tile_crs = map_array_ds.crs
    # tile_transform = map_array_ds.transform
    # tile_height = map_array_ds.height
    # tile_width = map_array_ds.width
    # sq_m_per_pixel = map_array_ds.transform[0] **2

    # map_array = map_array_ds.read()
    # map_array = np.transpose(map_array, (1, 2, 0)) # Transpose to (H, W, C)

    # # Convert basal area from square meters per pixel to square feet per acre
    # map_array[:, :, 0] *= 10.7639  # Convert basal area from square meters to square feet
    # map_array[:, :, 0] = (map_array[:, :, 0] / sq_m_per_pixel) * 4046.86  # sq ft per acre

    # # Convert max height from meters to feet
    # map_array[:, :, 1] = map_array[:, :, 1] * 3.28084

    # # Convert tree count from tree count per pixel to count per acre
    # map_array[:, :, 2] = (map_array[:, :, 2] / sq_m_per_pixel) * 4046.86

    # # Clip to 0 and replace outliers with nans
    # map_array[:, :, 0] = clip_tile(map_array[:, :, 0], MAX_BASAL_AREA)
    # map_array[:, :, 1] = clip_tile(map_array[:, :, 1], MAX_MAX_HT)
    # map_array[:, :, 2] = clip_tile(map_array[:, :, 2], MAX_TPA)
    # map_array[:, :, 3] = clip_tile(map_array[:, :, 3], 1.0)
    # map_array[:, :, 4] = clip_tile(map_array[:, :, 4], 1.0)
    # nan_mask = np.any(np.isnan(map_array[:, :, :3]), axis=-1)
    # map_array[nan_mask, :] = np.nan

    # map_array = np.transpose(map_array, (2, 0, 1))

    # with rasterio.open(
    #     output_file_path,
    #     'w',
    #     driver='GTiff',
    #     height=tile_height,
    #     width=tile_width,
    #     count=5,
    #     dtype=np.float32,
    #     crs=tile_crs,
    #     transform=tile_transform
    # ) as dst:
    #     dst.write(map_array)
    # exit()

    # # CONVERT UNETS TO FIAUNITS
    # unet_path = AOI_PATH / Path("./UNET-input")
    # tiles = list(unet_path.glob("*.tif"))

    # # For clipping. Max from FIA.
    # MAX_TPA = 1347 # Count per acre
    # MAX_MAX_HT = 191 # Feet
    # MAX_BASAL_AREA = 861 # feet^2 per acre

    # for tile in tqdm(tiles):
    #     output_file_path = AOI_PATH / f"./UNET-fia-units/{tile.stem}.tif"

    #     # Open and get info
    #     tile_ds = rasterio.open(tile)
    #     tile_crs = tile_ds.crs
    #     tile_transform = tile_ds.transform
    #     tile_height = tile_ds.height
    #     tile_width = tile_ds.width
    #     sq_m_per_pixel = tile_ds.transform[0] **2

    #     tile_array = tile_ds.read()
    #     tile_array = np.transpose(tile_array, (1, 2, 0)) # Transpose to (H, W, C)

    #     # Convert basal area from square meters per pixel to square feet per acre
    #     tile_array[:, :, 0] *= 10.7639  # Convert basal area from square meters to square feet
    #     tile_array[:, :, 0] = (tile_array[:, :, 0] / sq_m_per_pixel) * 4046.86  # sq ft per acre

    #     # Convert max height from meters to feet
    #     tile_array[:, :, 1] = tile_array[:, :, 1] * 3.28084

    #     # Convert tree count from tree count per pixel to count per acre
    #     tile_array[:, :, 2] = (tile_array[:, :, 2] / sq_m_per_pixel) * 4046.86

    #     # Clip to 0 and replace outliers with nans
    #     tile_array[:, :, 0] = clip_tile(tile_array[:, :, 0], MAX_BASAL_AREA)
    #     tile_array[:, :, 1] = clip_tile(tile_array[:, :, 1], MAX_MAX_HT)
    #     tile_array[:, :, 2] = clip_tile(tile_array[:, :, 2], MAX_TPA)
    #     tile_array[:, :, 3] = clip_tile(tile_array[:, :, 3], 1.0)
    #     tile_array[:, :, 4] = clip_tile(tile_array[:, :, 4], 1.0)
    #     nan_mask = np.any(np.isnan(tile_array[:, :, :3]), axis=-1)
    #     tile_array[nan_mask, :] = np.nan

    #     tile_array = np.transpose(tile_array, (2, 0, 1))

    #     with rasterio.open(
    #         output_file_path,
    #         'w',
    #         driver='GTiff',
    #         height=tile_height,
    #         width=tile_width,
    #         count=5,
    #         dtype=np.float32,
    #         crs=tile_crs,
    #         transform=tile_transform
    #     ) as dst:
    #         dst.write(tile_array)

    # exit()
