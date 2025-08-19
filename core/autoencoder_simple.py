"""
autoencoder_simple.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ForestDataset(Dataset):
    def __init__(self, features, plot_ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.plot_ids = torch.tensor(plot_ids, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.plot_ids[idx]


class CustomMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(CustomMSELoss, self).__init__()
        self.weights = weights  # Feature weights tensor

    def forward(self, inputs, targets):
        # Compute squared error
        sq_error = (inputs - targets) ** 2

        # Apply feature weights if provided
        if self.weights is not None:
            sq_error = sq_error * self.weights

        # Mean across all dimensions
        return torch.mean(sq_error)


class ForestAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, dropout_rate=0.2):
        super(ForestAutoencoder, self).__init__()

        # Encoder with dropout for regularization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

        # Decoder with dropout for regularization
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        # Get embedding
        embedding = self.encoder(x)
        # Reconstruct input
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)


def train_autoencoder(plot_data, feature_weights=None, embedding_dim=32, batch_size=64,
                      learning_rate=0.001, num_epochs=150, dropout_rate=0.2):
    """
    Train the autoencoder with improved parameters

    Args:
        plot_data: Polars DataFrame with plot data
        feature_weights: Dictionary mapping feature names to weight values
        embedding_dim: Dimensionality of the embedding space
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        dropout_rate: Dropout rate for regularization

    Returns:
        Trained model and related objects
    """
    # Define feature columns and target
    feature_cols = [
        'LIVE_CANOPY_CVR_PCT',
        'TPA_UNADJ',
        'MAX_HT',
        'BASAL_AREA_SUBP',
        'BALIVE',
        'ELEV',
        'SLOPE',
        'ASPECT_COS',
        'ASPECT_SIN',
    ]



    X = plot_data.select(feature_cols).to_numpy()
    plot_ids = plot_data.select(pl.col('SUBPLOTID')).to_numpy().flatten()

    # Check for NaN values and handle them
    if np.isnan(X).any():
        print(f"Warning: Found {np.isnan(X).sum()} NaN values in features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, plot_ids, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = ForestDataset(X_train, y_train)
    test_dataset = ForestDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = len(feature_cols)
    model = ForestAutoencoder(input_dim, embedding_dim=embedding_dim, dropout_rate=dropout_rate)

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    # Define loss function with feature weights
    if feature_weights is None:
        # Default weights emphasizing important metrics
        # Order: LIVE_CANOPY_CVR_PCT, TPA_UNADJ, MAX_HT, BALIVE, ELEV, SLOPE, ASPECT, FORTYPCD
        feature_weights = torch.tensor([1.0, 3.0, 2.0, 3.0, 1.0, 1.0, 0.5, 1.0], dtype=torch.float32)
    else:
        # Convert weights dictionary to tensor in the right order
        feature_weights = torch.tensor([feature_weights.get(col, 1.0) for col in feature_cols],
                                       dtype=torch.float32)

    criterion = CustomMSELoss(feature_weights)

    # Training loop
    num_epochs = num_epochs
    best_loss = float('inf')
    best_model = None
    patience = 500  # Early stopping patience
    patience_counter = 0

    # For plotting
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for features, _ in train_loader:
            # Forward pass
            reconstructed, _ = model(features)

            # Compute loss with weighted features
            loss = criterion(reconstructed, features)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, _ in test_loader:
                reconstructed, _ = model(features)
                loss = criterion(reconstructed, features)
                val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        # Store for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # Load best model
    model.load_state_dict(best_model)
    print(f'Training complete. Best validation loss: {best_loss:.4f}')

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')

    return model, scaler, X_test, y_test, plot_data, feature_cols


def find_most_similar_plots(model, scaler, X_test, y_test, plot_data, feature_cols, top_k=5):
    """
    For each test sample, find the most similar plots in the training set
    based on the learned embeddings.

    Fixed to exclude self-matches by ensuring we don't select the same plot ID.
    """
    model.eval()

    # Get all plot data
    X_all = plot_data.select(feature_cols).to_numpy()

    # Handle NaN values
    if np.isnan(X_all).any():
        print(f"Warning: Found {np.isnan(X_all).sum()} NaN values in features. Replacing with 0.")
        X_all = np.nan_to_num(X_all, nan=0.0)

    X_all_scaled = scaler.transform(X_all)

    # Use plot_ids array that aligns with X_all
    plot_ids_all = plot_data.select(['SUBPLOTID']).to_numpy().flatten()

    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)

    with torch.no_grad():
        # Get embeddings for all plots
        all_embeddings = model.get_embedding(X_all_tensor).numpy()

        # Get embeddings for test plots
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_embeddings = model.get_embedding(X_test_tensor).numpy()

    # Initialize results
    results = []

    # For each test plot
    for i, (test_embedding, true_plot_id) in enumerate(zip(test_embeddings, y_test)):
        # Calculate distances to all plots
        distances = np.sqrt(np.sum((all_embeddings - test_embedding) ** 2, axis=1))

        # Create a mask to exclude the exact same plot ID
        different_plot_mask = plot_ids_all != true_plot_id

        # Apply mask to get valid distances and plot IDs
        valid_distances = distances[different_plot_mask]
        valid_plot_ids = plot_ids_all[different_plot_mask]

        # Find indices of k most similar plots among the valid candidates
        if len(valid_distances) >= top_k:
            top_indices = np.argsort(valid_distances)[:top_k]

            # Get plot IDs and distances
            top_plot_ids = valid_plot_ids[top_indices]
            top_distances = valid_distances[top_indices]

            # Add to results
            result = {
                'test_idx': i,
                'true_plot_id': true_plot_id,
                'top_k_plots': [(plot_id, dist) for plot_id, dist in zip(top_plot_ids, top_distances)]
            }
            results.append(result)
        else:
            print(f"Warning: Not enough valid candidates for test sample {i} (Plot ID: {true_plot_id})")

    print(f"Processed {len(results)} test samples")
    return results


def visualize_embeddings(model, scaler, plot_data, feature_cols, save_path='embeddings_tsne.png'):
    """
    Visualize the learned embeddings using t-SNE colored by forest type
    """
    model.eval()

    # Get all plot data
    X = plot_data.select(feature_cols).to_numpy()

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Scale data
    X_scaled = scaler.transform(X)

    # Get forest types for coloring
    forest_types = plot_data.select(['FORTYPCD']).to_numpy().flatten()

    # Get embeddings
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.get_embedding(X_tensor).numpy()

    # Use t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a scatter plot of the embeddings
    plt.figure(figsize=(12, 10))

    # Define a color map based on forest types
    unique_forest_types = np.unique(forest_types)

    # Create a scatter plot
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=forest_types,
        cmap='tab20',
        alpha=0.7,
        s=30
    )

    plt.colorbar(scatter, label='Forest Type')
    plt.title('t-SNE Visualization of Forest Plot Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Embedding visualization saved to {save_path}")


def evaluate_predictions(results, plot_data, feature_cols):
    """
    Evaluate how well our approach works by comparing important metrics
    between the predicted plots and the true plots

    Fixed to calculate realistic RMSE values by using column indices
    """
    # Important metrics to compare
    key_metrics = ['BASAL_AREA_SUBP','BALIVE', 'MAX_HT', 'TPA_UNADJ']
    plot_data_dict = {}

    # Get column indices for faster lookup
    col_indices = {col: plot_data.columns.index(col) for col in plot_data.columns}

    # Convert plot_data to dictionary for faster lookup
    for i in range(len(plot_data)):
        row = plot_data.row(i)
        plot_id = row[col_indices['SUBPLOTID']]  # Use numeric index instead of string

        if plot_id not in plot_data_dict:
            plot_data_dict[plot_id] = {}

        for col in feature_cols:
            if col in col_indices:
                plot_data_dict[plot_id][col] = row[col_indices[col]]

    # Calculate errors for each test sample
    total_rmse = 0
    errors_by_metric = {metric: [] for metric in key_metrics}

    for result in results:
        true_plot_id = result['true_plot_id']
        # The first item in top_k_plots should be a different plot now
        predicted_plot_id = result['top_k_plots'][0][0]

        # Verify we're not comparing a plot to itself
        if true_plot_id == predicted_plot_id:
            print(f"WARNING: True plot ID {true_plot_id} matches predicted plot ID!")
            continue

        # If either plot doesn't exist in dictionary, skip
        if true_plot_id not in plot_data_dict or predicted_plot_id not in plot_data_dict:
            continue

        # Calculate errors for each key metric
        for metric in key_metrics:
            if metric not in plot_data_dict[true_plot_id] or metric not in plot_data_dict[predicted_plot_id]:
                continue

            true_value = plot_data_dict[true_plot_id][metric]
            pred_value = plot_data_dict[predicted_plot_id][metric]

            error = (true_value - pred_value) ** 2
            errors_by_metric[metric].append(error)

    # Calculate RMSE for each metric
    rmse_by_metric = {}
    for metric, errors in errors_by_metric.items():
        if errors:  # Check if list is not empty
            rmse = np.sqrt(np.mean(errors))
            rmse_by_metric[metric] = rmse
            print(f"RMSE for {metric}: {rmse:.4f}")
        else:
            print(f"Warning: No valid comparisons for {metric}")

    # Overall RMSE across key metrics
    all_errors = []
    for errors in errors_by_metric.values():
        all_errors.extend(errors)

    if all_errors:
        overall_rmse = np.sqrt(np.mean(all_errors))
        print(f"Overall RMSE for key metrics: {overall_rmse:.4f}")
    else:
        print("Warning: No valid comparisons for any metrics")

    return rmse_by_metric


def display_similarity_results(results, plot_data, feature_cols, num_examples=5):
    """
    Display detailed information about the similarity search results
    to verify the quality of matches.
    """
    import random

    # Get column indices for faster lookup
    col_indices = {col: plot_data.columns.index(col) for col in plot_data.columns}

    # Important metrics to focus on
    key_metrics = ['BASAL_AREA_SUBP','BALIVE', 'MAX_HT', 'TPA_UNADJ', 'LIVE_CANOPY_CVR_PCT', 'FORTYPCD']

    # Build a quick lookup dictionary for plot data
    plot_data_dict = {}
    for i in range(len(plot_data)):
        row = plot_data.row(i)
        plot_id = row[col_indices['SUBPLOTID']]

        if plot_id not in plot_data_dict:
            plot_data_dict[plot_id] = {}

        for col in key_metrics:
            idx = col_indices.get(col, None)
            if idx is not None:
                plot_data_dict[plot_id][col] = row[idx]

    # Select random examples to display
    if len(results) > num_examples:
        example_indices = random.sample(range(len(results)), num_examples)
    else:
        example_indices = range(len(results))

    # Create figure for bar charts
    fig, axes = plt.subplots(len(example_indices), 1, figsize=(12, 5 * len(example_indices)))
    if len(example_indices) == 1:
        axes = [axes]

    # Display results for each example
    for i, idx in enumerate(example_indices):
        result = results[idx]
        true_plot_id = result['true_plot_id']
        similar_plots = result['top_k_plots']

        print(f"\nExample {i + 1}:")
        print(f"True Plot ID: {true_plot_id}")

        # Get true plot metrics
        true_metrics = {}
        if true_plot_id in plot_data_dict:
            true_metrics = plot_data_dict[true_plot_id]
            print("True Plot Metrics:")
            for metric in key_metrics:
                if metric in true_metrics:
                    print(f"  {metric}: {true_metrics[metric]}")

        # Display top similar plots and their metrics
        print("\nTop Similar Plots:")
        similar_metrics_list = []

        for j, (similar_plot_id, distance) in enumerate(similar_plots[:3]):  # Show top 3
            print(f"{j + 1}. Plot ID: {similar_plot_id}, Distance: {distance:.6f}")

            if similar_plot_id in plot_data_dict:
                similar_metrics = plot_data_dict[similar_plot_id]
                similar_metrics_list.append(similar_metrics)

                for metric in key_metrics:
                    if metric in similar_metrics and metric in true_metrics:
                        true_val = true_metrics[metric]
                        sim_val = similar_metrics[metric]
                        diff = abs(true_val - sim_val)
                        percent_diff = (diff / true_val * 100) if true_val != 0 else float('inf')

                        print(f"  {metric}: {sim_val} (Diff: {diff:.2f}, {percent_diff:.2f}%)")

        # Create bar chart for visual comparison
        ax = axes[i]
        metrics_to_plot = [m for m in key_metrics if m in true_metrics]

        # Prepare bar positions
        x = np.arange(len(metrics_to_plot))
        bar_width = 0.2

        # Plot true values
        true_values = [true_metrics.get(m, 0) for m in metrics_to_plot]
        ax.bar(x, true_values, width=bar_width, label='True Plot', color='blue')

        # Plot similar plot values
        for j, similar_metrics in enumerate(similar_metrics_list[:3]):
            values = [similar_metrics.get(m, 0) for m in metrics_to_plot]
            ax.bar(x + (j + 1) * bar_width, values, width=bar_width,
                   label=f'Similar Plot {j + 1}', alpha=0.7)

        # Add labels and legend
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Values', fontsize=12)
        ax.set_title(f'Comparison of Plot Metrics (Example {i + 1})', fontsize=14)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()

        # Add percent differences as text
        for j, similar_metrics in enumerate(similar_metrics_list[:3]):
            for k, metric in enumerate(metrics_to_plot):
                if metric in similar_metrics and metric in true_metrics:
                    true_val = true_metrics[metric]
                    sim_val = similar_metrics[metric]
                    if true_val != 0:
                        percent_diff = (abs(true_val - sim_val) / true_val * 100)
                        ax.text(k + (j + 1) * bar_width, sim_val,
                                f"{percent_diff:.1f}%",
                                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('plot_comparisons.png', dpi=300, bbox_inches='tight')
    print("\nPlot comparison chart saved as 'plot_comparisons.png'")

    # Calculate overall statistics
    print("\nOverall Statistics:")
    total_comparisons = 0
    metric_diffs = {metric: [] for metric in key_metrics}

    for result in results:
        true_plot_id = result['true_plot_id']
        if true_plot_id not in plot_data_dict:
            continue

        similar_plot_id = result['top_k_plots'][0][0]  # Best match
        if similar_plot_id not in plot_data_dict:
            continue

        total_comparisons += 1

        for metric in key_metrics:
            if (metric in plot_data_dict[true_plot_id] and
                    metric in plot_data_dict[similar_plot_id]):
                true_val = plot_data_dict[true_plot_id][metric]
                sim_val = plot_data_dict[similar_plot_id][metric]

                if true_val != 0:
                    percent_diff = (abs(true_val - sim_val) / abs(true_val)) * 100
                    metric_diffs[metric].append(percent_diff)

    # Report median percent differences
    print(f"Total valid comparisons: {total_comparisons}")
    for metric in key_metrics:
        if metric_diffs[metric]:
            median_diff = np.median(metric_diffs[metric])
            mean_diff = np.mean(metric_diffs[metric])
            print(f"{metric} - Median percent difference: {median_diff:.2f}%, Mean: {mean_diff:.2f}%")

    return fig


def main():
    # Load data
    from DataFrames import create_polars_dataframe_by_subplot, create_polars_dataframe_by_plot  # Import your data loading function
    plot_data = create_polars_dataframe_by_subplot()

    # Define custom feature weights (optional)
    feature_weights = {
        'BASAL_AREA_SUBP': 4.0,
        'BALIVE': 4.0,  # Most important
        'TPA_UNADJ': 3.0,  # Very important
        'MAX_HT': 3.0,  # Very important
        'LIVE_CANOPY_CVR_PCT': 2.0,  # Important
        'FORTYPCD': 1.5,  # Somewhat important
        'ELEV': 1.0,  # Standard importance
        'SLOPE': 1.0,  # Standard importance
        'ASPECT': 0.5  # Less important
    }

    # Train autoencoder with improved parameters
    model, scaler, X_test, y_test, plot_data, feature_cols = train_autoencoder(
        plot_data,
        feature_weights=feature_weights,
        embedding_dim=64,  # Larger embedding dimension
        batch_size=128,  # Larger batch size
        learning_rate=0.001,
        num_epochs=200,  # More epochs with early stopping
        dropout_rate=0.3  # Increased dropout for regularization
    )

    # Visualize embeddings
    visualize_embeddings(model, scaler, plot_data, feature_cols)

    # Find similar plots (fixed to exclude self-matches)
    results = find_most_similar_plots(model, scaler, X_test, y_test, plot_data, feature_cols, top_k=5)

    # Display detailed results for a few examples
    display_similarity_results(results, plot_data, feature_cols, num_examples=3)

    # Evaluate predictions
    rmse_by_metric = evaluate_predictions(results, plot_data, feature_cols)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_weights': feature_weights
    }, 'autoencoder_simple.pt')

    print("Model saved as autoencoder_simple.pt")


if __name__ == "__main__":
    main()