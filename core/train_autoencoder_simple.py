"""
train_autoencoder_simple.py

Main script to train and evaluate the Simple Autoencoder.
"""
import torch
import polars as pl
from DataFrames import create_polars_dataframe_by_subplot

# Import shared modules
from utils import prepare_data
from models import SimpleAutoencoder
from train import train_autoencoder
from evaluation import find_most_similar_plots, evaluate_predictions
from visualization import visualize_embeddings, display_similarity_results


def main():
    # --- Configuration for Simple Autoencoder ---
    MODEL_NAME = 'simple_autoencoder'
    PLOT_ID_COL = 'SUBPLOTID'

    # Feature columns used by this model
    feature_cols = [
        "TREE_COUNT",
        'MAX_HT',
        'AVG_HT',
        'BASAL_AREA_TREE',
        'ELEV',
        'SLOPE',
        'ASPECT_COS',
        'ASPECT_SIN',
        "LAT",
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
    # Key metrics for evaluation
    key_metrics = ["TREE_COUNT", 'MAX_HT', 'BASAL_AREA_TREE', "QMD_TREE", 'AVG_HT', "FORTYPCD"]

    # Custom feature weights
    feature_weights = {
    }

    # Model and Training parameters
    embedding_dim = 64
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 200
    dropout_rate = 0.3

    # --- Data Loading and Preparation ---
    print("Loading data...")
    plot_data = create_polars_dataframe_by_subplot()

    print("Preparing data loaders...")
    train_loader, test_loader, scaler, X_test, y_test = prepare_data(
        plot_data, feature_cols, PLOT_ID_COL, batch_size=batch_size
    )

    # --- Model Initialization ---
    print(f"Initializing {MODEL_NAME}...")
    input_dim = len(feature_cols)
    model = SimpleAutoencoder(input_dim, embedding_dim=embedding_dim, dropout_rate=dropout_rate)

    # --- Training ---
    print("Starting training...")
    model, _, _ = train_autoencoder(
        model, train_loader, test_loader, feature_cols,
        feature_weights=feature_weights, learning_rate=learning_rate, num_epochs=num_epochs
    )

    # --- Evaluation and Visualization ---
    print("Starting evaluation and visualization...")
    visualize_embeddings(
        model, scaler, plot_data, feature_cols,
        color_by_col='FORTYPCD', save_path=f'{MODEL_NAME}_tsne.png'
    )

    results = find_most_similar_plots(
        model, scaler, plot_data, feature_cols, PLOT_ID_COL, X_test, y_test
    )

    display_similarity_results(results, plot_data, PLOT_ID_COL, key_metrics, num_examples=3)

    evaluate_predictions(results, plot_data, PLOT_ID_COL, key_metrics)

    # --- Save Model ---
    print("Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_weights': feature_weights
    }, f'{MODEL_NAME}.pt')
    print(f"Model saved as {MODEL_NAME}.pt")


if __name__ == "__main__":
    main()