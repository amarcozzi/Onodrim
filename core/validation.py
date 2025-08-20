"""
validation.py

This script:
1. Loads the trained model and TWO datasets:
   a. The original training data (FIA data)
   b. The validation data (from a different source)
2. Embeds both datasets in the latent space
3. For each validation plot, finds the closest training plot
4. Computes differences between validation plots and best matching training plots
5. Visualizes and reports accuracy metrics
6. Creates a raster of the imputed FIA plot IDs.
"""

import os
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm
import polars as pl
import warnings
import rasterio
from models import AttentionAutoencoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

plots_dir_default = "./validation"
data_dir_default = "./data"
core_dir_default = "."


def load_model(model_path):
    """Load the trained model with PyTorch compatibility"""
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals(['sklearn.preprocessing._data.StandardScaler'])
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except (ImportError, AttributeError, RuntimeError):
        print("Warning: Failed to load model with add_safe_globals. Trying weights_only=False.")
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            print(f"Error: Could not load model using any method. {e}")
            raise

    model_params = checkpoint['model_params']
    feature_cols = checkpoint['feature_cols']
    scaler = checkpoint['scaler']

    use_attention = checkpoint.get('use_attention', model_params.get('use_attention', False))
    if not use_attention and 'evaluation_metrics' in checkpoint and 'model_name' in checkpoint.get('evaluation_metrics',
                                                                                                   {}):
        model_name = checkpoint['evaluation_metrics']['model_name']
        use_attention = 'attention' in model_name.lower()

    model = AttentionAutoencoder(
        input_dim=model_params['input_dim'],
        latent_dim=model_params['latent_dim'],
        hidden_dims=model_params['hidden_dims'],
        dropout_rate=model_params['dropout_rate'],
        use_attention=use_attention
    )

    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}. Model weights might be random.")

    model.eval()
    return model, scaler, feature_cols


def load_training_data(fia_data_path):
    """Load the original training data (FIA data)"""
    try:
        training_data = pl.read_csv(fia_data_path)
        print(f"Loaded {len(training_data)} training plots from FIA data at {fia_data_path}")
        return training_data
    except Exception as e:
        print(f"Error loading training data from {fia_data_path}: {e}")
        return None


def load_validation_data(geojson_path):
    """Load validation data from the provided GeoJSON"""
    try:
        validation_data = gpd.read_file(geojson_path)
        print(f"Loaded {len(validation_data)} validation plots from {geojson_path}")
        print(f"Validation data columns: {list(validation_data.columns)}")
        return validation_data
    except Exception as e:
        print(f"Error loading validation data from {geojson_path}: {e}")
        return None


def embed_dataset(model, data, feature_cols, scaler, description="Dataset"):
    """Generate embeddings for a dataset using the model"""
    print(f"Generating embeddings for {description}...")

    available_features = [col for col in feature_cols if col in data.columns]
    print(f"Found {len(available_features)}/{len(feature_cols)} features in {description}")

    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        print(
            f"Warning: Missing features in {description}: {missing_features}. These will be filled with 0 before scaling.")

    df_for_embedding = pd.DataFrame(0, index=np.arange(len(data)), columns=feature_cols)
    for col in available_features:
        if isinstance(data, pl.DataFrame):
            df_for_embedding[col] = data[col].fill_null(0).to_numpy()
        else:
            df_for_embedding[col] = data[col].fillna(0).to_numpy()

    X = df_for_embedding.to_numpy()
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.get_embedding(X_tensor).numpy()

    print(f"Generated embeddings for {description} with shape: {embeddings.shape}")
    return embeddings, X_scaled


def run_cross_dataset_validation(model, scaler, training_data, validation_data, feature_cols, plots_dir_path):
    """Run validation by comparing validation data against training data"""
    key_metrics = ['TREE_COUNT', 'BASAL_AREA_TREE', 'MAX_HT', 'QMD_TREE']

    train_metrics_available = [m for m in key_metrics if m in training_data.columns]
    val_metrics_available = [m for m in key_metrics if m in validation_data.columns]
    available_metrics = [m for m in key_metrics if m in train_metrics_available and m in val_metrics_available]

    print(f"Metrics available in both datasets for comparison: {available_metrics}")
    if not available_metrics:
        print("Critical error: No common metrics found to compare!")
        return None, None

    train_embeddings, _ = embed_dataset(model, training_data, feature_cols, scaler, "training data")
    val_embeddings, _ = embed_dataset(model, validation_data, feature_cols, scaler, "validation data")

    results = []
    train_df_pandas = training_data.to_pandas() if isinstance(training_data, pl.DataFrame) else training_data

    print("Finding most similar training plots for each validation plot...")
    for i in tqdm(range(len(validation_data)), desc="Processing validation plots"):
        distances = np.sqrt(np.sum((train_embeddings - val_embeddings[i]) ** 2, axis=1))
        distances = np.nan_to_num(distances, nan=np.inf)
        closest_idx = int(np.argmin(distances))

        result = {'val_idx': i, 'train_idx': closest_idx, 'distance': float(distances[closest_idx])}
        for metric in available_metrics:
            val_value = validation_data[metric].iloc[i]  # Use .iloc for GeoDataFrame
            train_value = train_df_pandas[metric].iloc[closest_idx]
            result[f'val_{metric}'] = val_value
            result[f'train_{metric}'] = train_value
            result[f'error_{metric}'] = train_value - val_value
            if pd.notna(val_value) and val_value != 0:
                result[f'pct_error_{metric}'] = abs(result[f'error_{metric}'] / val_value) * 100
            else:
                result[f'pct_error_{metric}'] = np.nan
        results.append(result)
    results_df = pd.DataFrame(results)

    metrics_stats = {}
    print("\n===== Cross-Dataset Validation Results =====")
    print(f"Validation performed on {len(results_df)} plots.")
    avg_distance = results_df['distance'].mean()
    median_distance = results_df['distance'].median()
    print(f"Embedding Distances: Mean={avg_distance:.4f}, Median={median_distance:.4f}")

    print("\n--- Detailed Metrics ---")
    for metric in available_metrics:
        val_values = results_df[f'val_{metric}'].dropna()
        train_values = results_df.loc[val_values.index, f'train_{metric}'].dropna()
        common_indices = val_values.index.intersection(train_values.index)
        val_values = val_values.loc[common_indices]
        train_values = train_values.loc[common_indices]

        if len(val_values) == 0:
            metrics_stats[metric] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
                                     'mean_pct_error': np.nan, 'median_pct_error': np.nan}
            continue
        current_pct_errors = results_df.loc[common_indices, f'pct_error_{metric}'].dropna()
        stats = {
            'rmse': np.sqrt(mean_squared_error(val_values, train_values)),
            'mae': mean_absolute_error(val_values, train_values),
            'r2': r2_score(val_values, train_values),
            'mean_pct_error': current_pct_errors.mean(),
            'median_pct_error': current_pct_errors.median()
        }
        metrics_stats[metric] = stats
        print(f"\nMetric: {metric} (based on {len(val_values)} non-NA pairs)")
        print(f"  R²: {stats['r2']:.4f}, RMSE: {stats['rmse']:.4f}, MAE: {stats['mae']:.4f}")
        print(f"  Mean Abs % Err: {stats['mean_pct_error']:.2f}%, Median Abs % Err: {stats['median_pct_error']:.2f}%")

    create_validation_visualizations(results_df, metrics_stats, available_metrics, plots_dir_path)
    return results_df, metrics_stats


def create_validation_visualizations(results_df, metrics_stats, available_metrics, plots_dir_path):
    print("Creating improved validation visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['distance'].dropna(), kde=True, bins=50)
    plt.title('Distribution of Embedding Distances', fontsize=14)
    plt.xlabel('Euclidean Distance', fontsize=12);
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(os.path.join(plots_dir_path, 'embedding_distances_distribution.png'));
    plt.close()

    for metric in available_metrics:
        if pd.isna(metrics_stats[metric]['r2']): continue
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'Cross-Validation Analysis for {metric}', fontsize=18)

        ax1 = axes[0, 0]
        sns.histplot(results_df[f'val_{metric}'].dropna(), color='skyblue', alpha=0.7, label='Validation (Actual)',
                     kde=True, stat="density", ax=ax1)
        sns.histplot(results_df[f'train_{metric}'].dropna(), color='salmon', alpha=0.7, label='Best Match (Predicted)',
                     kde=True, stat="density", ax=ax1)
        ax1.set_title(f'Distribution Comparison', fontsize=14);
        ax1.set_xlabel(metric, fontsize=12);
        ax1.set_ylabel('Density', fontsize=12);
        ax1.legend()

        ax2 = axes[0, 1]
        val_m = results_df[f'val_{metric}'].dropna();
        train_m = results_df.loc[val_m.index, f'train_{metric}'].dropna()
        common_idx = val_m.index.intersection(train_m.index)
        if len(results_df) > 1000 and len(common_idx) > 100:
            hb = ax2.hexbin(val_m[common_idx], train_m[common_idx], gridsize=50, cmap='Blues', mincnt=1)
            fig.colorbar(hb, ax=ax2, label='Counts')
        else:
            ax2.scatter(val_m[common_idx], train_m[common_idx], alpha=0.4, s=15, color='steelblue')
        min_v = min(val_m[common_idx].min(), train_m[common_idx].min());
        max_v = max(val_m[common_idx].max(), train_m[common_idx].max())
        ax2.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Ideal')
        stats_txt = (
            f"R²={metrics_stats[metric]['r2']:.3f}\nRMSE={metrics_stats[metric]['rmse']:.3f}\nMAE={metrics_stats[metric]['mae']:.3f}")
        ax2.text(0.05, 0.95, stats_txt, transform=ax2.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax2.set_title(f'Actual vs. Predicted', fontsize=14);
        ax2.set_xlabel(f'Actual {metric}', fontsize=12);
        ax2.set_ylabel(f'Predicted {metric}', fontsize=12);
        ax2.legend();
        ax2.grid(True)

        ax3 = axes[1, 0];
        errors = (results_df[f'train_{metric}'] - results_df[f'val_{metric}']).dropna()
        sns.histplot(errors, kde=True, ax=ax3, bins=50, color='forestgreen')
        ax3.axvline(errors.mean(), color='r', ls='--', label=f'Mean Err: {errors.mean():.2f}')
        ax3.axvline(errors.median(), color='purple', ls=':', label=f'Median Err: {errors.median():.2f}')
        ax3.set_title(f'Error Distribution (Pred - Actual)', fontsize=14);
        ax3.set_xlabel('Error', fontsize=12);
        ax3.set_ylabel('Frequency', fontsize=12);
        ax3.legend()

        ax4 = axes[1, 1];
        pct_err = results_df[f'pct_error_{metric}'].dropna()
        cap = np.percentile(pct_err[pct_err <= 1000], 99) if len(pct_err[pct_err <= 1000]) > 0 else 200;
        cap = max(cap, 50)
        sns.histplot(pct_err.clip(0, cap), kde=True, ax=ax4, bins=50, color='darkorange')
        ax4.axvline(metrics_stats[metric]['mean_pct_error'], color='r', ls='--',
                    label=f"Mean: {metrics_stats[metric]['mean_pct_error']:.1f}%")
        ax4.axvline(metrics_stats[metric]['median_pct_error'], color='purple', ls=':',
                    label=f"Median: {metrics_stats[metric]['median_pct_error']:.1f}%")
        ax4.set_title(f'Abs. Percent Error (capped at {cap:.0f}%)', fontsize=14);
        ax4.set_xlabel('Abs. Percent Error (%)', fontsize=12);
        ax4.set_ylabel('Frequency', fontsize=12);
        ax4.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
        plt.savefig(os.path.join(plots_dir_path, f'cross_validation_analysis_{metric}.png'));
        plt.close(fig)

    plt.figure(figsize=(12, 8));
    palette = sns.color_palette("viridis", len(available_metrics))
    for i, metric in enumerate(available_metrics):
        if pd.isna(metrics_stats[metric]['r2']): continue
        abs_err = (results_df[f'train_{metric}'] - results_df[f'val_{metric}']).abs().dropna()
        dist_m = results_df.loc[abs_err.index, 'distance']
        err_cap = np.percentile(abs_err, 98) if len(abs_err) > 0 else abs_err.max();
        err_cap = max(err_cap, 1e-6)
        plt.scatter(dist_m, abs_err.clip(upper=err_cap), alpha=0.3, label=f'{metric} (Abs Err, cap @{err_cap:.2f})',
                    color=palette[i], s=20)
    plt.xlabel('Embedding Distance', fontsize=12);
    plt.ylabel('Absolute Error (Capped)', fontsize=12)
    plt.title('Embedding Distance vs. Absolute Prediction Error', fontsize=15);
    plt.legend(loc='best', fontsize=10);
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(plots_dir_path, 'distance_vs_absolute_error.png'));
    plt.close()

    plt.figure(figsize=(12, 7));
    metrics_plot = [m for m in available_metrics if not pd.isna(metrics_stats[m]['r2'])]
    r2_vals = [metrics_stats[m]['r2'] for m in metrics_plot];
    colors = ['skyblue' if r2 >= 0 else 'salmon' for r2 in r2_vals]
    bars = plt.bar(metrics_plot, r2_vals, color=colors);
    plt.ylabel('R-squared (R²)', fontsize=12);
    plt.title('R-squared by Metric', fontsize=15);
    plt.axhline(0, color='k', lw=0.8, ls='--')
    for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}',
                                                       va='bottom' if yval >= 0 else 'top', ha='center', fontsize=10)
    plt.xticks(rotation=45, ha="right");
    plt.tight_layout();
    plt.savefig(os.path.join(plots_dir_path, 'cross_validation_r2_summary.png'));
    plt.close()

    plt.figure(figsize=(12, 7));
    median_pct_err_vals = [metrics_stats[m]['median_pct_error'] for m in metrics_plot]
    bars = plt.bar(metrics_plot, median_pct_err_vals, color='mediumpurple');
    plt.ylabel('Median Abs Percent Error (%)', fontsize=12);
    plt.title('Median Abs Percent Error by Metric', fontsize=15)
    for bar in bars: yval = bar.get_height(); plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}%',
                                                       va='bottom', ha='center', fontsize=10)
    plt.xticks(rotation=45, ha="right");
    plt.tight_layout();
    plt.savefig(os.path.join(plots_dir_path, 'cross_validation_median_pct_error_summary.png'));
    plt.close()
    print(f"All visualizations saved to {plots_dir_path}")


def create_imputation_raster(results_df, validation_data, training_data, template_raster_path, output_raster_path,
                             nodata_value=-9999):
    """
    Creates a raster where each pixel value is the SUBPLOTID of the imputed FIA plot.
    """
    print(f"Creating imputation raster at {output_raster_path}...")

    try:
        with rasterio.open(template_raster_path) as src:
            meta = src.meta.copy()
            template_height = src.height
            template_width = src.width
    except Exception as e:
        print(f"Error reading template raster {template_raster_path}: {e}")
        return

    train_df_pandas = training_data.to_pandas() if isinstance(training_data, pl.DataFrame) else training_data

    id_col_name = None
    if 'SUBPLOTID' in train_df_pandas.columns:
        id_col_name = 'SUBPLOTID'
    elif 'PLT_CN' in train_df_pandas.columns:  # Fallback, less ideal
        id_col_name = 'PLT_CN'
        print("Warning: 'SUBPLOTID' not found in training data. Using 'PLT_CN' as identifier for raster.")

    if not id_col_name:
        print("Error: No suitable ID column ('SUBPLOTID' or 'PLT_CN') found in training data. Cannot create raster.")
        return

    # Ensure the ID column is numeric (int64) for the raster
    if not pd.api.types.is_numeric_dtype(train_df_pandas[id_col_name]) or \
            not np.issubdtype(train_df_pandas[id_col_name].dtype, np.integer):
        print(
            f"Warning: ID column '{id_col_name}' (dtype: {train_df_pandas[id_col_name].dtype}) in training_data is not int64. Attempting conversion.")
        original_type = train_df_pandas[id_col_name].dtype
        train_df_pandas[id_col_name] = pd.to_numeric(train_df_pandas[id_col_name], errors='coerce')
        failed_conversions = train_df_pandas[id_col_name].isna().sum()
        if failed_conversions > 0:
            print(f"Warning: {failed_conversions} '{id_col_name}' values could not be converted to numeric.")
        train_df_pandas[id_col_name] = train_df_pandas[id_col_name].fillna(nodata_value).astype(np.int64)
        print(
            f"'{id_col_name}' converted from {original_type} to {train_df_pandas[id_col_name].dtype}, with NaNs/errors replaced by {nodata_value}.")

    output_array = np.full((template_height, template_width), nodata_value, dtype=np.int64)
    meta.update(dtype='int64', nodata=nodata_value, count=1)

    if 'row' not in validation_data.columns or 'col' not in validation_data.columns:
        print("Error: 'row' and 'col' columns missing in validation_data GeoDataFrame. Cannot map to raster pixels.")
        return

    num_pixels_set = 0
    for _, result_row in tqdm(results_df.iterrows(), total=results_df.shape[0], desc="Populating imputation raster"):
        val_original_idx = result_row['val_idx']
        try:
            # 'row' and 'col' from validation_data are 0-indexed from bottom-left as per inventory.py's get_tree_row_col
            gdf_r = int(validation_data.iloc[val_original_idx]['row'])
            gdf_c = int(validation_data.iloc[val_original_idx]['col'])

            # Raster row index is 0-indexed from top-left
            raster_r_remapped = template_height - 1 - gdf_r

            if not (0 <= raster_r_remapped < template_height and 0 <= gdf_c < template_width):
                continue  # Pixel out of bounds

            train_match_idx = result_row['train_idx']
            imputed_plot_id = train_df_pandas.iloc[train_match_idx][id_col_name]

            if pd.isna(imputed_plot_id) or imputed_plot_id == nodata_value:  # Check if ID is valid after conversion
                output_array[raster_r_remapped, gdf_c] = nodata_value
            else:
                output_array[raster_r_remapped, gdf_c] = imputed_plot_id
                num_pixels_set += 1
        except (KeyError, IndexError):
            # This might happen if val_idx or train_idx are out of bounds for their respective dataframes
            # Or if 'row'/'col' are missing for a specific val_idx
            continue
        except Exception as e_assign:
            print(f"Error assigning value for val_idx {val_original_idx}: {e_assign}. Assigning NoData if possible.")
            if 'raster_r_remapped' in locals() and 'gdf_c' in locals() and \
                    (0 <= raster_r_remapped < template_height and 0 <= gdf_c < template_width):
                output_array[raster_r_remapped, gdf_c] = nodata_value

    print(f"Set {num_pixels_set} pixels in the imputation raster out of {results_df.shape[0]} processed results.")
    if num_pixels_set == 0 and results_df.shape[0] > 0:
        print("Warning: No pixels were set. Check 'row'/'col' mapping, data integrity, and remapping logic.")

    try:
        with rasterio.open(output_raster_path, 'w', **meta) as dst:
            dst.write(output_array, 1)
        print(f"Imputation raster saved successfully to {output_raster_path}")
    except Exception as e:
        print(f"Error writing output raster {output_raster_path}: {e}")


def main():
    """Main function to run the cross-dataset validation"""
    print("Starting cross-dataset forest model validation...")

    base_data_dir = data_dir_default  # Use the global default or make it configurable
    model_filename = "weights/attention_autoencoder.pt"
    model_path = os.path.join(core_dir_default, model_filename)

    training_data_filename = "output.csv"
    training_data_path = os.path.join(base_data_dir, training_data_filename)

    validation_data_filename = "summary.geojson"
    validation_data_path = os.path.join(base_data_dir, validation_data_filename)

    output_plots_dir = plots_dir_default
    os.makedirs(output_plots_dir, exist_ok=True)  # Ensure plots_dir exists
    os.makedirs(base_data_dir, exist_ok=True)  # Ensure data_dir exists

    output_csv_path = os.path.join(base_data_dir, 'cross_validation_results_improved.csv')

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        # Basic fallback to current directory for model
        alt_model_path = model_filename
        if os.path.exists(alt_model_path):
            print(f"Found model at {alt_model_path}, using this instead.")
            model_path = alt_model_path
        else:
            return

    try:
        model, scaler, feature_cols = load_model(model_path)
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        return

    training_data = load_training_data(training_data_path)
    if training_data is None: return

    validation_data = load_validation_data(validation_data_path)
    if validation_data is None: return

    results_df, metrics_stats = run_cross_dataset_validation(
        model, scaler, training_data, validation_data, feature_cols, output_plots_dir
    )

    if results_df is not None:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Cross-validation results saved to {output_csv_path}")

        # Create the imputation raster
        template_raster_path = os.path.join(base_data_dir, "summary.tif")
        imputation_raster_path = os.path.join(base_data_dir, "fia_imputation_raster.tif")

        if os.path.exists(template_raster_path):
            create_imputation_raster(
                results_df,
                validation_data,
                training_data,
                template_raster_path,
                imputation_raster_path
            )
        else:
            print(f"Warning: Template raster {template_raster_path} not found. Cannot create imputation raster.")

    print("Cross-dataset validation complete!")


if __name__ == "__main__":
    import sys  # Required for the check in main for autoencoder_attention

    main()