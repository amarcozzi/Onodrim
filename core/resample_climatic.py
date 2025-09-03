import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rioxarray
import xarray
from shapely.geometry import box

AOI_PATH = Path("./inference/coconino")
SHAPE_PATH = Path("./data/coconino_shape.geojson")
CLIMATIC_PATH = Path("./data/climatic")
OUTPUT_PATH = AOI_PATH / Path("./climatic-interpolated")

def main():
    gdf = gpd.read_file(SHAPE_PATH)

    climatic_files = sorted(list(CLIMATIC_PATH.glob("*.tif")))

    for climatic_file in tqdm(climatic_files):

        file_name = f"{climatic_file.stem}.tif"
        output_file_path = OUTPUT_PATH / file_name
        if output_file_path.exists():
            print(f"{file_name} already exists, skipping.")
            continue
        print(f"Creating interpolated raster for {file_name}, will save to {output_file_path}")

        cf = rioxarray.open_rasterio(climatic_file)

        # Clipping to bounding box based on shape file
        res_x, res_y = cf.rio.resolution()
        minx, miny, maxx, maxy = gdf.total_bounds
        minx -= res_x
        maxx += res_x
        miny += res_y
        maxy -= res_y
        rect_geom = [box(minx, miny, maxx, maxy)]
        cf_clipped = cf.rio.clip(rect_geom, gdf.crs, drop=True)

        # Getting original min and max values
        orig_min = cf_clipped.min().item()
        orig_max = cf_clipped.max().item()

        # Creating x and y points for resampled raster
        xmin, ymin, xmax, ymax = cf_clipped.rio.bounds()

        res = 0.0001 # About 10m
        x_new = np.arange(xmin, xmax, res)
        y_new = np.arange(ymax, ymin, -res)
        clipped_sorted = cf_clipped.sortby("y")

        # Bilinearly interpolating
        resampled = clipped_sorted.interp(
            x=x_new,
            y=y_new,
            method="cubic"
        )

        # Reversing y again
        resampled = resampled.sortby("y", ascending=False)

        # Clipping values to original min and max
        resampled = resampled.clip(min=orig_min, max=orig_max)

        # Make sure nodata and crs are right
        if cf.rio.nodata is not None:
            resampled = resampled.rio.write_nodata(cf.rio.nodata, encoded=True)
        resampled = resampled.rio.write_crs(cf.rio.crs, inplace=False)

        # Write out to output
        resampled.rio.to_raster(output_file_path, dtype=cf_clipped.dtype)

if __name__ == "__main__":
    main()