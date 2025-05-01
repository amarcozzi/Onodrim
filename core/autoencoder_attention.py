import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


class SelfAttention(nn.Module):
    """Self-attention layer for tabular data features"""

    def __init__(self, input_dim, attention_dim=None):
        super(SelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // 2

        # Projection layers for queries, keys, values
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))

    def forward(self, x):
        # Input shape: [batch_size, feature_dim]
        # We need to add a "sequence length" dimension for attention
        # Treating each feature as a "sequence" position
        x_reshaped = x.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Project to queries, keys, values
        q = self.query(x_reshaped)  # [batch_size, 1, attention_dim]
        k = self.key(x_reshaped)  # [batch_size, 1, attention_dim]
        v = self.value(x_reshaped)  # [batch_size, 1, feature_dim]

        # Calculate attention scores
        # Attention shape: [batch_size, 1, 1]
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention, dim=-1)

        # Apply attention to values
        # Output shape: [batch_size, 1, feature_dim]
        context = torch.matmul(attention_weights, v)

        # Reshape back to original dimensions
        output = context.squeeze(1)  # [batch_size, feature_dim]

        # Residual connection
        return output + x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            # nn.BatchNorm1d(num_features=out_features),
        )

        # If dimensions don't match, use a linear projection
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        return self.relu(x)


class ForestAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=(128, 96), dropout_rate=0.2, use_attention=True):
        super(ForestAutoencoder, self).__init__()

        # Build encoder with residual blocks and attention
        encoder_layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            encoder_layers.append(ResidualBlock(prev_dim, dim))
            encoder_layers.append(nn.Dropout(dropout_rate))

            # Add attention after each residual block
            if use_attention:
                encoder_layers.append(SelfAttention(dim))

            prev_dim = dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder with symmetry to encoder
        decoder_layers = []
        prev_dim = latent_dim

        for dim in reversed(hidden_dims):
            decoder_layers.append(ResidualBlock(prev_dim, dim))
            decoder_layers.append(nn.Dropout(dropout_rate))

            # Add attention in decoder too
            if use_attention:
                decoder_layers.append(SelfAttention(dim))

            prev_dim = dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Get embedding
        embedding = self.encode(x)
        # Reconstruct input
        reconstruction = self.decode(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encode(x)

def create_output_directories():
    """Create directories for outputs"""
    plots_dir = "./plots"
    data_dir = "./data"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    return plots_dir, data_dir


def train_autoencoder(plot_data, feature_cols, feature_weights=None, latent_dim=32, hidden_dims=(128, 96),
                      batch_size=64, learning_rate=0.001, num_epochs=150, dropout_rate=0.2, use_attention=True):
    """
    Train the residual autoencoder with improved parameters

    Args:
        plot_data: Polars DataFrame with plot data
        feature_weights: Dictionary mapping feature names to weight values
        latent_dim: Dimensionality of the latent space (embedding)
        hidden_dims: Tuple of hidden dimensions for residual blocks
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        dropout_rate: Dropout rate for regularization

    Returns:
        Trained model and related objects
    """
    # Process input data
    X = plot_data.select(feature_cols).to_numpy()
    plot_ids = plot_data.select(['SUBPLOTID']).to_numpy().flatten()

    # Handle NaN values
    if np.isnan(X).any():
        print(f"Warning: Found {np.isnan(X).sum()} NaN values in features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, plot_ids, test_size=0.2, random_state=42)
    print(f"There are {len(X_train)} training samples and {len(X_test)} test samples")

    # Create datasets and dataloaders
    train_dataset = ForestDataset(X_train, y_train)
    test_dataset = ForestDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = len(feature_cols)
    model = ForestAutoencoder(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate, use_attention=use_attention)

    # Print model details
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print(f"Model architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Dropout rate: {dropout_rate}")

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

    # Define loss function with feature weights
    if feature_weights is None:
        # Default weights emphasizing important metrics
        default_weights = []
        for col in feature_cols:
            if col == 'BASAL_AREA_TREE':
                default_weights.append(4.0)
            elif col == 'TREE_COUNT':
                default_weights.append(3.0)
            elif col == 'MAX_HT':
                default_weights.append(3.0)
            elif col == 'QMD_TREE':
                default_weights.append(2.0)
            elif col == 'FORTYPCD':
                default_weights.append(1.5)
            else:
                default_weights.append(1.0)

        feature_weights = torch.tensor(default_weights, dtype=torch.float32)
    else:
        # Convert weights dictionary to tensor in the right order
        feature_weights = torch.tensor([feature_weights.get(col, 1.0) for col in feature_cols],
                                       dtype=torch.float32)

    criterion = CustomMSELoss(feature_weights)

    # Training loop
    best_loss = float('inf')
    best_model = None
    patience = 30  # Early stopping patience
    patience_counter = 0

    # For plotting
    train_losses = []
    val_losses = []

    # Setup directories for outputs
    plots_dir, _ = create_output_directories()

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

        # Early stopping if validation loss exceeds training loss
        if avg_val_loss > avg_train_loss and epoch > 200:  # Allow some initial epochs
            print(f'Early stopping at epoch {epoch + 1} due to validation loss exceeding training loss')
            break

        # # Early stopping based on patience
        # if patience_counter >= patience:
        #     print(f'Early stopping at epoch {epoch + 1} due to no improvement')
        #     break

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
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'training_curve.png'))
    plt.close()

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
    and other key metrics
    """
    plots_dir, _ = create_output_directories()
    save_path = os.path.join(plots_dir, save_path)

    model.eval()

    # Get all plot data
    X = plot_data.select(feature_cols).to_numpy()

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Scale data
    X_scaled = scaler.transform(X)

    # Get key metrics for coloring
    forest_types = plot_data.select(['FORTYPCD']).to_numpy().flatten()
    balive = plot_data.select(['BASAL_AREA_TREE']).to_numpy().flatten()
    max_ht = plot_data.select(['MAX_HT']).to_numpy().flatten()
    avg_ht = plot_data.select(['AVG_HT']).to_numpy().flatten()
    tree_count = plot_data.select(['TREE_COUNT']).to_numpy().flatten()
    qmd_tree = plot_data.select(['QMD_TREE']).to_numpy().flatten()

    # Get embeddings
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.get_embedding(X_tensor).numpy()

    # Use t-SNE to reduce to 2D for visualization
    print("Computing t-SNE embedding (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create multiple visualizations
    metrics = {
        'forest_type': {
            'values': forest_types,
            'title': 'Forest Type (FORTYPCD)',
            'cmap': 'tab20',
        },
        'balive': {
            'values': balive,
            'title': 'Basal Area Live (BALIVE)',
            'cmap': 'viridis',
        },
        'max_ht': {
            'values': max_ht,
            'title': 'Max Height (MAX_HT)',
            'cmap': 'plasma',
        },
        'avg_ht': {
            'values': avg_ht,
            'title': 'Avg Height (AVG_HT)',
            'cmap': 'plasma',
        },
        'tree_count': {
            'values': tree_count,
            'title': 'Tree Count',
            'cmap': 'inferno',
        },
        'qmd_tree': {
            'values': qmd_tree,
            'title': 'QMD of tree stems ( DIA >= 5)',
            'cmap': 'cividis',
        }
    }

    # Create a separate plot for each metric
    for key, metric_info in metrics.items():
        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=metric_info['values'],
            cmap=metric_info['cmap'],
            alpha=0.7,
            s=30
        )

        plt.colorbar(scatter, label=metric_info['title'])
        plt.title(f't-SNE Visualization Colored by {metric_info["title"]}', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=14)
        plt.ylabel('t-SNE Dimension 2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)

        save_name = f'embeddings_tsne_{key}.png'
        plt.savefig(os.path.join(plots_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_name}")

    # Create a combined 2x2 visualization for the main metrics
    plt.figure(figsize=(20, 16))

    # Forest type (top left)
    plt.subplot(2, 2, 1)
    scatter1 = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=forest_types,
        cmap='tab20',
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter1, label='Forest Type (FORTYPCD)')
    plt.title('Colored by Forest Type', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # BALIVE (top right)
    plt.subplot(2, 2, 2)
    scatter2 = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=balive,
        cmap='viridis',
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter2, label='BASAL_AREA_TREE')
    plt.title('Colored by Basal Area of Tree Stems', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # MAX_HT (bottom left)
    plt.subplot(2, 2, 3)
    scatter3 = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=max_ht,
        cmap='plasma',
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter3, label='MAX_HT')
    plt.title('Colored by Max Height', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Canopy cover (bottom right)
    plt.subplot(2, 2, 4)
    scatter4 = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=qmd_tree,
        cmap='cividis',
        alpha=0.7,
        s=30
    )
    plt.colorbar(scatter4, label='QMD_TREE')
    plt.title('Colored by QMD', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('t-SNE Visualization of Forest Plot Embeddings', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, 'embeddings_tsne_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Embedding visualizations saved to {plots_dir}")

    return embeddings_2d


def visualize_metric_performance(predictions_df, metric, plots_dir, model_name):
    """
    Create a comprehensive visualization for a single metric's performance

    Args:
        predictions_df: DataFrame with prediction results
        metric: The metric to visualize
        plots_dir: Directory to save plots
        model_name: Name of the model for plot titles
    """
    # Create a figure with subplots
    if metric != 'FORTYPCD':
        # For numeric metrics, use a 2x2 grid
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Actual vs Predicted Scatter Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(
            predictions_df[f'true_{metric}'],
            predictions_df[f'pred_{metric}'],
            alpha=0.5
        )

        # Add reference line
        min_val = min(predictions_df[f'true_{metric}'].min(), predictions_df[f'pred_{metric}'].min())
        max_val = max(predictions_df[f'true_{metric}'].max(), predictions_df[f'pred_{metric}'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Calculate statistics for title
        rmse = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
        r2 = 1 - (np.sum(predictions_df[f'error_{metric}'] ** 2) /
                  np.sum((predictions_df[f'true_{metric}'] - predictions_df[f'true_{metric}'].mean()) ** 2))

        ax1.set_title(f'Actual vs Predicted {metric}\nRMSE: {rmse:.4f}, R²: {r2:.4f}')
        ax1.set_xlabel(f'Actual {metric}')
        ax1.set_ylabel(f'Predicted {metric}')
        ax1.grid(True, alpha=0.3)

        # 2. Error Distribution Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(
            predictions_df[f'error_{metric}'],
            kde=True,
            bins=30,
            ax=ax2
        )
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title(f'Error Distribution for {metric}\nMean error: {predictions_df[f"error_{metric}"].mean():.4f}')
        ax2.set_xlabel(f'Error (Predicted - Actual)')
        ax2.set_ylabel('Frequency')

        # 3. Percent Error Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        # Cap percent errors at 100% for better visualization
        capped_pct_errors = predictions_df[f'pct_error_{metric}'].clip(0, 100)
        sns.histplot(
            capped_pct_errors,
            kde=True,
            bins=30,
            ax=ax3
        )
        ax3.set_title(f'Percent Error Distribution for {metric}\nMedian: {capped_pct_errors.median():.2f}%')
        ax3.set_xlabel('Percent Error')
        ax3.set_ylabel('Frequency')

        # 4. Error vs Actual Value
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(
            predictions_df[f'true_{metric}'],
            predictions_df[f'error_{metric}'],
            alpha=0.5
        )
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title(f'Error vs Actual {metric}')
        ax4.set_xlabel(f'Actual {metric}')
        ax4.set_ylabel('Error (Predicted - Actual)')
        ax4.grid(True, alpha=0.3)

    else:
        # For categorical metrics like FORTYPCD, create a confusion matrix
        fig = plt.figure(figsize=(12, 10))

        # Get unique forest type codes
        unique_codes = sorted(set(predictions_df[f'true_{metric}'].unique()) |
                              set(predictions_df[f'pred_{metric}'].unique()))

        # Create confusion matrix
        conf_matrix = pd.crosstab(
            predictions_df[f'true_{metric}'],
            predictions_df[f'pred_{metric}'],
            rownames=['Actual'],
            colnames=['Predicted'],
            normalize='index'  # Normalize by row (actual values)
        )

        # Plot confusion matrix
        ax = fig.add_subplot(111)
        sns.heatmap(
            conf_matrix,
            annot=True,
            cmap='Blues',
            fmt='.2f',
            ax=ax
        )
        ax.set_title(f'Confusion Matrix for {metric}')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_{metric.lower()}_analysis.png'))
    plt.close(fig)


def visualize_errors_by_forest_type(predictions_df, metric, plots_dir, model_name):
    """
    Create visualization of errors broken down by forest type

    Args:
        predictions_df: DataFrame with prediction results
        metric: The metric to visualize (should be numeric)
        plots_dir: Directory to save plots
        model_name: Name of the model for plot titles
    """
    if metric == 'FORTYPCD':
        return  # Skip for FORTYPCD itself

    # Create a figure
    fig = plt.figure(figsize=(15, 10))

    # 1. Boxplot of errors by forest type
    ax1 = fig.add_subplot(121)

    # Group by true forest type
    forest_types = sorted(predictions_df['true_FORTYPCD'].unique())
    error_by_forest = {}

    for ft in forest_types:
        mask = predictions_df['true_FORTYPCD'] == ft
        if mask.sum() > 0:  # Only include if we have data
            error_by_forest[ft] = predictions_df.loc[mask, f'error_{metric}']

    # Sort forest types by median error
    sorted_types = sorted(
        error_by_forest.items(),
        key=lambda x: x[1].median() if len(x[1]) > 0 else 0
    )
    sorted_type_ids = [x[0] for x in sorted_types]

    # Create boxplot
    boxplot_data = [error_by_forest[ft] for ft in sorted_type_ids if ft in error_by_forest]
    boxplot_labels = [str(int(ft)) for ft in sorted_type_ids if ft in error_by_forest]

    if boxplot_data:  # Only plot if we have data
        ax1.boxplot(boxplot_data, labels=boxplot_labels, vert=True)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title(f'Error Distribution by Forest Type ({metric})')
        ax1.set_xlabel('Forest Type Code')
        ax1.set_ylabel(f'Error in {metric}')
        ax1.tick_params(axis='x', rotation=90)
        ax1.grid(True, axis='y', alpha=0.3)

    # 2. Barplot of RMSE by forest type
    ax2 = fig.add_subplot(122)

    # Calculate RMSE for each forest type
    rmse_by_forest = {}
    counts_by_forest = {}

    for ft in forest_types:
        mask = predictions_df['true_FORTYPCD'] == ft
        if mask.sum() >= 5:  # Only include if we have enough data
            errors = predictions_df.loc[mask, f'error_{metric}']
            rmse_by_forest[ft] = np.sqrt(np.mean(errors ** 2))
            counts_by_forest[ft] = len(errors)

    # Sort by RMSE
    sorted_rmse = sorted(rmse_by_forest.items(), key=lambda x: x[1])
    sorted_types = [x[0] for x in sorted_rmse]
    sorted_rmses = [x[1] for x in sorted_rmse]
    sorted_counts = [counts_by_forest[ft] for ft in sorted_types]

    if sorted_types:  # Only plot if we have data
        bars = ax2.bar(range(len(sorted_types)), sorted_rmses, alpha=0.7)

        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, sorted_counts)):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'n={count}',
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90
            )

        # Add overall RMSE reference line
        overall_rmse = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
        ax2.axhline(y=overall_rmse, color='r', linestyle='--',
                    label=f'Overall RMSE: {overall_rmse:.2f}')

        ax2.set_title(f'RMSE by Forest Type ({metric})')
        ax2.set_xlabel('Forest Type Code')
        ax2.set_ylabel(f'RMSE of {metric}')
        ax2.set_xticks(range(len(sorted_types)))
        ax2.set_xticklabels([str(int(ft)) for ft in sorted_types])
        ax2.tick_params(axis='x', rotation=90)
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_{metric.lower()}_by_forest_type.png'))
    plt.close(fig)


def visualize_prediction_results(model, scaler, X_test, y_test, plot_data, feature_cols,
                                 plots_dir="./plots", model_name="ResidualAutoencoder", top_k=3):
    """
    Comprehensive visualization of model results with detailed error analysis

    Args:
        model: Trained autoencoder model
        scaler: Feature scaler used for preprocessing
        X_test: Test features
        y_test: Test plot IDs
        plot_data: Original plot data DataFrame
        feature_cols: List of feature columns
        plots_dir: Directory to save plots
        model_name: Name of the model for plot titles
        top_k: Number of similar plots to find
    """
    # Find most similar plots for test data
    similar_plots_results = find_most_similar_plots(model, scaler, X_test, y_test, plot_data, feature_cols, top_k)

    # Create a lookup dictionary for faster access to plot data
    plot_data_dict = {}
    for i in range(len(plot_data)):
        row = plot_data.row(i)
        plot_id = row[plot_data.columns.index('SUBPLOTID')]

        if plot_id not in plot_data_dict:
            plot_data_dict[plot_id] = {}

        for col in plot_data.columns:
            col_idx = plot_data.columns.index(col)
            plot_data_dict[plot_id][col] = row[col_idx]

    # Key metrics for analysis
    key_metrics = ['BASAL_AREA_TREE', 'MAX_HT', 'AVG_HT', 'TREE_COUNT', "QMD_TREE", 'FORTYPCD']

    # Prepare data for analysis
    predictions = []

    for result in similar_plots_results:
        true_plot_id = result['true_plot_id']
        predicted_plot_id = result['top_k_plots'][0][0]  # Top match

        # Skip if either plot is missing from our dictionary
        if true_plot_id not in plot_data_dict or predicted_plot_id not in plot_data_dict:
            continue

        # Extract values for all key metrics
        pred_row = {
            'true_plot_id': true_plot_id,
            'predicted_plot_id': predicted_plot_id,
            'distance': result['top_k_plots'][0][1]  # Embedding distance
        }

        for metric in key_metrics:
            if metric in plot_data_dict[true_plot_id] and metric in plot_data_dict[predicted_plot_id]:
                pred_row[f'true_{metric}'] = plot_data_dict[true_plot_id][metric]
                pred_row[f'pred_{metric}'] = plot_data_dict[predicted_plot_id][metric]

                # Calculate errors
                error = plot_data_dict[predicted_plot_id][metric] - plot_data_dict[true_plot_id][metric]
                abs_error = abs(error)

                # Avoid division by zero for percent error
                true_val = plot_data_dict[true_plot_id][metric]
                if true_val != 0:
                    pct_error = (abs_error / abs(true_val)) * 100
                else:
                    pct_error = np.nan

                pred_row[f'error_{metric}'] = error
                pred_row[f'abs_error_{metric}'] = abs_error
                pred_row[f'pct_error_{metric}'] = pct_error

        predictions.append(pred_row)

    # Convert to DataFrame for easier analysis
    predictions_df = pd.DataFrame(predictions)

    # Print overall statistics
    print(f"\n===== {model_name} Performance Analysis =====")
    print(f"Total valid predictions: {len(predictions_df)}")

    # Visualize each metric separately
    for metric in key_metrics:
        # Skip forest type code (FORTYPCD) for some numeric analyses
        is_numeric = metric != 'FORTYPCD'

        if is_numeric:
            # Calculate key statistics
            rmse = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
            mae = predictions_df[f'abs_error_{metric}'].mean()
            median_pct_error = predictions_df[f'pct_error_{metric}'].median()
            mean_pct_error = predictions_df[f'pct_error_{metric}'].mean()

            print(f"\n{metric} Statistics:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Median Percent Error: {median_pct_error:.2f}%")
            print(f"  Mean Percent Error: {mean_pct_error:.2f}%")
        else:
            # For categorical variables like FORTYPCD
            accuracy = (predictions_df[f'true_{metric}'] == predictions_df[f'pred_{metric}']).mean() * 100
            print(f"\n{metric} Statistics:")
            print(f"  Accuracy: {accuracy:.2f}%")

        # Create a comprehensive figure for each metric
        visualize_metric_performance(predictions_df, metric, plots_dir, model_name)

        # Create error by forest type plot (for non-FORTYPCD metrics)
        if metric != 'FORTYPCD':
            visualize_errors_by_forest_type(predictions_df, metric, plots_dir, model_name)

    # Correlation between embedding distance and prediction error
    plt.figure(figsize=(10, 6))
    for metric in [m for m in key_metrics if m != 'FORTYPCD']:
        plt.scatter(
            predictions_df['distance'],
            predictions_df[f'abs_error_{metric}'],
            alpha=0.5,
            label=metric
        )

    plt.xlabel('Embedding Distance')
    plt.ylabel('Absolute Error')
    plt.title('Relationship Between Embedding Distance and Prediction Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_distance_vs_error.png'))

    # Display detailed example comparisons
    display_similarity_examples(similar_plots_results, plot_data_dict, key_metrics, plots_dir, model_name,
                                num_examples=5)

    return predictions_df


def display_similarity_examples(results, plot_data_dict, key_metrics, plots_dir, model_name, num_examples=5):
    """
    Display and visualize detailed examples of similarity matches

    Args:
        results: Results from find_most_similar_plots
        plot_data_dict: Dictionary with plot data
        key_metrics: List of key metrics to compare
        plots_dir: Directory to save plots
        model_name: Name of the model for plot titles
        num_examples: Number of examples to display
    """
    import random

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
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_example_comparisons.png'),
                dpi=300, bbox_inches='tight')
    print(f"\nExample comparison chart saved as '{model_name.lower().replace(' ', '_')}_example_comparisons.png'")


def evaluate_comprehensive(model, scaler, X_test, y_test, plot_data, feature_cols, model_name="ResidualAutoencoder"):
    """
    Comprehensive evaluation of the autoencoder model

    Args:
        model: Trained autoencoder model
        scaler: Feature scaler
        X_test: Test features
        y_test: Test plot IDs
        plot_data: Original plot data
        feature_cols: Feature columns
        model_name: Name of the model for outputs

    Returns:
        Dictionary with evaluation metrics
    """
    # Setup output directories
    plots_dir, data_dir = create_output_directories()

    # Get embedding visualization
    embeddings_2d = visualize_embeddings(model, scaler, plot_data, feature_cols)

    # Find similar plots and analyze errors
    predictions_df = visualize_prediction_results(
        model, scaler, X_test, y_test, plot_data,
        feature_cols, plots_dir, model_name
    )

    # Calculate metrics for key variables
    key_metrics = ['BASAL_AREA_TREE', 'MAX_HT', 'AVG_HT', 'TREE_COUNT', 'QMD_TREE', 'FORTYPCD',]

    # Store metrics
    metrics = {}
    for metric in key_metrics:
        if metric != 'FORTYPCD':  # Skip FORTYPCD for numeric metrics
            metrics[f'{metric}_rmse'] = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
            metrics[f'{metric}_mae'] = predictions_df[f'abs_error_{metric}'].mean()
            metrics[f'{metric}_median_pct_error'] = predictions_df[f'pct_error_{metric}'].median()
        else:
            # For forest type, calculate accuracy
            metrics[f'{metric}_accuracy'] = (predictions_df[f'true_{metric}'] ==
                                             predictions_df[f'pred_{metric}']).mean() * 100

    # Create a summary bar plot for RMSE
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract RMSE values for numeric metrics
    rmse_metrics = [m for m in key_metrics if m != 'FORTYPCD']
    rmse_values = [metrics[f'{m}_rmse'] for m in rmse_metrics]

    # Create bar chart
    bars = ax.bar(rmse_metrics, rmse_values)

    # Add labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f'{height:.2f}',
            ha='center',
            va='bottom'
        )

    ax.set_title(f'{model_name} RMSE by Metric')
    ax.set_xlabel('Metric')
    ax.set_ylabel('RMSE')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name.lower().replace(" ", "_")}_rmse_summary.png'))

    # Print overall summary
    print("\n===== OVERALL MODEL PERFORMANCE =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return {
        'metrics': metrics,
        'embeddings': embeddings_2d,
        'predictions': predictions_df
    }


def main():
    """Main function to run the entire pipeline"""

    # Create output directories
    plots_dir, data_dir = create_output_directories()

    # Load data
    from DataFrames import create_polars_dataframe_by_subplot
    plot_data = create_polars_dataframe_by_subplot()

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
        'FORTYPCD',
        "QMD_TREE",
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


    # Define custom feature weights
    feature_weights = {
        'BASAL_AREA_TREE': 4.0,  # Most important
        'TREE_COUNT': 3.0,  # Very important,
        "QMD_TREE": 4.0,  # Very important
        'MAX_HT': 3.0,  # Very important
        'AVG_HT': 3.0,
        'FORTYPCD': 5.0,  # Important for forest type matching
        'ELEV': 1.0,  # Standard importance
        'SLOPE': 1.0,  # Standard importance
        'ASPECT_SIN': 0.5,  # Lower importance (derived feature)
        'ASPECT_COS': 0.5,  # Lower importance (derived feature)
        'LAT': 0.25,  # Location information,

    }

    # Train residual autoencoder
    latent_dim = 16
    hidden_dims = [32]
    dropout_rate = 0.2
    model, scaler, X_test, y_test, plot_data, feature_cols = train_autoencoder(
        plot_data,
        feature_cols,
        feature_weights=feature_weights,
        latent_dim=latent_dim,  # Latent dimension size
        hidden_dims=hidden_dims,  # Residual block dimensions
        batch_size=128,
        learning_rate=0.001,
        num_epochs=300,  # Allow longer training with early stopping
        dropout_rate=0.2,
        use_attention=False
    )

    # Comprehensive evaluation
    evaluation = evaluate_comprehensive(
        model, scaler, X_test, y_test, plot_data, feature_cols,
        model_name="AttentionAutoencoder"
    )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feature_cols,
        'feature_weights': feature_weights,
        'model_params': {
            'input_dim': len(feature_cols),
            'latent_dim': 8,
            'hidden_dims': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
            'dropout_rate': 0.3
        },
        'evaluation_metrics': evaluation['metrics']
    }, os.path.join(data_dir, 'attention_autoencoder_model.pt'))

    print(f"Model saved as {os.path.join(data_dir, 'attention_autoencoder_model.pt')}")
    print(f"All visualizations saved to {plots_dir}")


if __name__ == "__main__":
    main()