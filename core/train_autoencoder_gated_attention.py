"""
train_autoencoder_gated_attention.py

Main script to train and evaluate the autoencoder model with a gated attention mechanism.
"""
import torch
from DataFrames import create_polars_dataframe_by_subplot
from pathlib import Path

# Import shared modules
from dataloader import prepare_data, create_output_directories
from models import AttentionAutoencoder, GatedAttention
from train_denoise import train_autoencoder
from evaluation import run_evaluation

AOI_PATH = Path("./inference/coconino")
STATE = "AZ"

def main():
    # --- Configuration for Attention Autoencoder ---
    MODEL_NAME = 'gated_attention_autoencoder'
    PLOT_ID_COL = 'SUBPLOTID'


    save_dir = AOI_PATH / "./weights" #f'./weights/{MODEL_NAME}.pt'
    if not save_dir.exists():
        print(f"Save path {save_dir} does not exist, exiting.")
        exit()

    # --- Setup ---
    create_output_directories()

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

    # Define continuous and categorical metrics for evaluation
    key_metrics_con = ['BASAL_AREA_TREE', 'MAX_HT', "TREE_COUNT", "GINI_DIA", "GINI_HT"]
    key_metrics_cat = ['FORTYPCD']

    # Custom feature weights
    feature_weights = {
    }

    # Model and Training parameters
    latent_dim = 16
    hidden_dims = [64]
    batch_size = 256
    learning_rate = 5e-5
    num_epochs = 1000
    dropout_rate = 0.2

    # --- Data Loading and Preparation ---
    print("Loading data...")
    plot_data = create_polars_dataframe_by_subplot(STATE, climate_resolution="10m")

    print("Preparing data loaders...")
    train_loader, test_loader, scaler, X_test, y_test = prepare_data(
        plot_data, feature_cols, PLOT_ID_COL, batch_size=batch_size
    )

    # --- Model Initialization ---
    print(f"Initializing {MODEL_NAME}...")
    input_dim = len(feature_cols)
    model = AttentionAutoencoder(
        input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
        dropout_rate=dropout_rate, attention_module=GatedAttention
    )

    # --- Training ---
    print("Starting training...")
    model, _, _ = train_autoencoder(
        model, MODEL_NAME, train_loader, test_loader, feature_cols,
        feature_weights=feature_weights, learning_rate=learning_rate, num_epochs=num_epochs, patience=100
    )

    # --- Evaluation and Visualization ---
    print("Starting evaluation and visualization...")
    evaluation_metrics = run_evaluation(
        model, scaler, plot_data, feature_cols, PLOT_ID_COL, X_test, y_test,
        key_metrics_con, key_metrics_cat, MODEL_NAME
    )

    # --- Save Model ---
    save_path = AOI_PATH / f'./weights/{MODEL_NAME}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'latent_dim': latent_dim,
        'hidden_dims': hidden_dims,
        'dropout_rate': dropout_rate,
        'attention_module': 'GatedAttention',
        'model_name': MODEL_NAME,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'evaluation_metrics': evaluation_metrics
    }, save_path)
    print(f"Model for {STATE} and scaler saved to {save_path}")


if __name__ == "__main__":
    main()