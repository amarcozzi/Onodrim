"""
visualization.py
"""
import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec


def visualize_embeddings(model, scaler, plot_data, feature_cols, metrics_to_color, model_name):
    """
    Visualizes embeddings using both t-SNE and UMAP, colored by various metrics.
    Saves all plots to the 'plots/' directory.
    """
    model.eval()
    X = plot_data.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.get_embedding(X_tensor).numpy()

    # --- Reducers ---
    reducers = {}
    reducers['tsne'] = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    reducers['umap'] = UMAP(n_components=2, random_state=42)

    for reducer_name, reducer in reducers.items():
        print(f"Computing {reducer_name.upper()} embeddings...")
        embeddings_2d = reducer.fit_transform(embeddings)

        # --- Plotting ---
        for metric in metrics_to_color:
            plt.figure(figsize=(12, 10))
            color_values = plot_data.select(metric).to_numpy().flatten()

            scatter = plt.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=color_values, cmap='viridis', alpha=0.7, s=30
            )

            plt.colorbar(scatter, label=metric)
            plt.title(f'{reducer_name.upper()} Visualization of Embeddings by {metric}', fontsize=16)
            plt.xlabel(f'{reducer_name.upper()} Dimension 1', fontsize=14)
            plt.ylabel(f'{reducer_name.upper()} Dimension 2', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)

            save_path = os.path.join('plots', f'{model_name}_{reducer_name}_{metric.lower()}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {reducer_name.upper()} plot to {save_path}")


def visualize_metric_performance(predictions_df, metric, is_categorical, model_name):
    """
    Creates a comprehensive visualization for a single metric's performance.
    Handles both continuous (2x2 grid) and categorical (confusion matrix) metrics.
    """
    fig_path = os.path.join('plots', f'{model_name}_{metric.lower()}_analysis.png')

    if not is_categorical:
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Actual vs Predicted Scatter Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(predictions_df[f'true_{metric}'], predictions_df[f'pred_{metric}'], alpha=0.5)
        min_val = min(predictions_df[f'true_{metric}'].min(), predictions_df[f'pred_{metric}'].min())
        max_val = max(predictions_df[f'true_{metric}'].max(), predictions_df[f'pred_{metric}'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        rmse = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
        ax1.set_title(f'Actual vs Predicted {metric}\nRMSE: {rmse:.4f}')
        ax1.set_xlabel(f'Actual {metric}'); ax1.set_ylabel(f'Predicted {metric}'); ax1.grid(True, alpha=0.3)

        # 2. Error Distribution Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(predictions_df[f'error_{metric}'], kde=True, bins=30, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title(f'Error Distribution for {metric}')
        ax2.set_xlabel('Error (Predicted - Actual)'); ax2.set_ylabel('Frequency')

        # 3. Percent Error Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        capped_pct_errors = predictions_df[f'pct_error_{metric}'].clip(-100, 100)
        sns.histplot(capped_pct_errors, kde=True, bins=30, ax=ax3)
        ax3.set_title(f'Percent Error Distribution for {metric}\nMedian: {capped_pct_errors.median():.2f}%')
        ax3.set_xlabel('Percent Error'); ax3.set_ylabel('Frequency')

        # 4. Error vs Actual Value
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(predictions_df[f'true_{metric}'], predictions_df[f'error_{metric}'], alpha=0.5)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title(f'Error vs Actual {metric}')
        ax4.set_xlabel(f'Actual {metric}'); ax4.set_ylabel('Error (Predicted - Actual)'); ax4.grid(True, alpha=0.3)
    else:
        # For categorical metrics, create a confusion matrix
        fig = plt.figure(figsize=(12, 10))
        conf_matrix = pd.crosstab(
            predictions_df[f'true_{metric}'], predictions_df[f'pred_{metric}'],
            rownames=['Actual'], colnames=['Predicted'], normalize='index'
        )
        ax = fig.add_subplot(111)
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f', ax=ax)
        ax.set_title(f'Confusion Matrix for {metric}')

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved metric performance plot to {fig_path}")

def display_similarity_examples(results, plot_data, plot_id_col, key_metrics, model_name, num_examples=5):
    """
    Displays and visualizes detailed examples of similarity matches with bar charts.
    """
    plot_data_pd = plot_data.to_pandas().set_index(plot_id_col)
    example_indices = random.sample(range(len(results)), min(num_examples, len(results)))

    fig, axes = plt.subplots(len(example_indices), 1, figsize=(12, 6 * len(example_indices)))
    if len(example_indices) == 1: axes = [axes]

    for i, idx in enumerate(example_indices):
        result = results[example_indices[i]]
        true_plot_id = result['true_plot_id']
        similar_plots = result['top_k_plots']

        ax = axes[i]
        true_metrics = plot_data_pd.loc[true_plot_id]
        metrics_to_plot = [m for m in key_metrics if m in true_metrics and not isinstance(true_metrics[m], str)]

        x = np.arange(len(metrics_to_plot))
        bar_width = 0.2

        # Plot true values
        true_values = [true_metrics.get(m, 0) for m in metrics_to_plot]
        ax.bar(x, true_values, width=bar_width, label='True Plot', color='blue')

        # Plot similar plot values
        for j, (sim_id, _) in enumerate(similar_plots[:3]):
            sim_metrics = plot_data_pd.loc[sim_id]
            values = [sim_metrics.get(m, 0) for m in metrics_to_plot]
            ax.bar(x + (j + 1) * bar_width, values, width=bar_width, label=f'Similar Plot {j+1}', alpha=0.7)

        ax.set_title(f'Metric Comparison for True Plot ID: {true_plot_id}', fontsize=14)
        ax.set_xticks(x + bar_width); ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
        ax.set_ylabel('Values'); ax.legend()

    fig_path = os.path.join('plots', f'{model_name}_example_comparisons.png')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Saved example comparison plot to {fig_path}")