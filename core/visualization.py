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
from networkx.algorithms.bipartite.basic import density
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from matplotlib.gridspec import GridSpec


# Define units for key metrics
UNITS = {
    'BASAL_AREA_TREE': 'ft²',
    'MAX_HT': 'ft',
    'AVG_HT': 'ft',
    'TREE_COUNT': None,
    'FORTYPCD': None  # Categorical, no units
}


def calculate_regression_metrics(true_values, pred_values):
    """Calculate RMSE, MAE, and R² for regression metrics."""
    rmse = np.sqrt(np.mean((pred_values - true_values) ** 2))
    mae = np.mean(np.abs(pred_values - true_values))
    r2 = r2_score(true_values, pred_values)
    return rmse, mae, r2


def calculate_classification_metrics(true_values, pred_values):
    """Calculate comprehensive classification metrics."""
    # Overall accuracy
    accuracy = (true_values == pred_values).mean()

    # Get classification report for precision, recall, f1-score
    report = classification_report(true_values, pred_values, output_dict=True, zero_division=0)

    # Extract macro averages
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'full_report': report
    }


def visualize_embeddings(model, scaler, plot_data, feature_cols, metrics_to_color_con, metrics_to_color_cat, model_name):
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
    reducers['tsne'] = TSNE(n_components=2, perplexity=30, max_iter=1000)
    reducers['umap'] = UMAP(n_components=2)

    for reducer_name, reducer in reducers.items():
        print(f"Computing {reducer_name.upper()} embeddings...")
        embeddings_2d = reducer.fit_transform(embeddings)

        metrics_to_color = metrics_to_color_con + metrics_to_color_cat

        # --- Plotting ---
        for metric in metrics_to_color:
            plt.figure(figsize=(12, 10))
            color_values = plot_data.select(metric).to_numpy().flatten()

            cmap = 'viridis' if metric in metrics_to_color_con else 'tab20'

            scatter = plt.scatter(
                embeddings_2d[:, 0], embeddings_2d[:, 1],
                c=color_values, cmap=cmap, alpha=0.7, s=30
            )

            # Add units to colorbar label if available
            units = UNITS.get(metric, '')
            colorbar_label = f"{metric} ({units})" if units else metric
            plt.colorbar(scatter, label=colorbar_label)

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
    units = UNITS.get(metric, '')

    if not is_categorical:
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(2, 2, figure=fig)

        # Calculate all regression metrics
        true_vals = predictions_df[f'true_{metric}']
        pred_vals = predictions_df[f'pred_{metric}']
        rmse, mae, r2 = calculate_regression_metrics(true_vals, pred_vals)

        # 1. Actual vs Predicted Scatter Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(true_vals, pred_vals, alpha=0.5)
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Enhanced title with all metrics and units
        title_text = f'Actual vs Predicted {metric}'
        if units:
            title_text += f' ({units})'
        title_text += f'\nRMSE: {rmse:.4f}'
        if units:
            title_text += f' {units}'
        title_text += f' | MAE: {mae:.4f}'
        if units:
            title_text += f' {units}'
        title_text += f' | R²: {r2:.4f}'

        ax1.set_title(title_text)

        xlabel = f'Actual {metric}'
        ylabel = f'Predicted {metric}'
        if units:
            xlabel += f' ({units})'
            ylabel += f' ({units})'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.grid(True, alpha=0.3)

        # 2. Error Distribution Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        errors = predictions_df[f'error_{metric}']
        sns.histplot(errors, kde=True, bins=30, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        title_text = f'Error Distribution for {metric}'
        ax2.set_title(title_text)
        xlabel = 'Error (Predicted - Actual)'
        if units:
            xlabel += f' ({units})'
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Frequency')

        # 3. Percent Error Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        capped_pct_errors = predictions_df[f'pct_error_{metric}'].clip(-100, 100)
        sns.histplot(capped_pct_errors, kde=True, bins=30, ax=ax3, )
        ax3.set_title(f'Percent Error Distribution for {metric}')
        ax3.set_xlabel('Percent Error (%)')
        ax3.set_ylabel('Frequency')

        # 4. Error vs Actual Value
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.scatter(true_vals, errors, alpha=0.5)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_title(f'Error vs Actual {metric}')
        xlabel = f'Actual {metric}'
        ylabel = 'Error (Predicted - Actual)'
        if units:
            xlabel += f' ({units})'
            ylabel += f' ({units})'
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(ylabel)
        ax4.grid(True, alpha=0.3)

    else:
        # For categorical metrics, create enhanced confusion matrix with metrics
        fig = plt.figure(figsize=(15, 12))

        # Calculate classification metrics
        true_vals = predictions_df[f'true_{metric}']
        pred_vals = predictions_df[f'pred_{metric}']
        metrics_dict = calculate_classification_metrics(true_vals, pred_vals)

        # Create confusion matrix (normalized)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1])

        # Main confusion matrix
        ax1 = fig.add_subplot(gs[0, :])
        conf_matrix = pd.crosstab(
            true_vals, pred_vals,
            rownames=['Actual'], colnames=['Predicted'], normalize='index'
        )
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f', ax=ax1)

        # Enhanced title with metrics
        title_text = f'Confusion Matrix for {metric}\n'
        title_text += f'Accuracy: {metrics_dict["accuracy"]:.3f} | '
        title_text += f'Macro Precision: {metrics_dict["macro_precision"]:.3f} | '
        title_text += f'Macro Recall: {metrics_dict["macro_recall"]:.3f} | '
        title_text += f'Macro F1: {metrics_dict["macro_f1"]:.3f}'
        ax1.set_title(title_text)

        # Metrics summary table
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')

        # Create summary text
        summary_text = "Classification Metrics Summary:\n"
        summary_text += f"Overall Accuracy: {metrics_dict['accuracy']:.4f}\n"
        summary_text += f"Macro Precision: {metrics_dict['macro_precision']:.4f}\n"
        summary_text += f"Macro Recall: {metrics_dict['macro_recall']:.4f}\n"
        summary_text += f"Macro F1-Score: {metrics_dict['macro_f1']:.4f}\n"
        summary_text += f"Total Samples: {len(true_vals)}"

        ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved metric performance plot to {fig_path}")

    # Return metrics for logging
    if not is_categorical:
        return {'rmse': rmse, 'mae': mae, 'r2': r2}
    else:
        return metrics_dict


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

        # Add units to x-axis labels
        labels_with_units = []
        for metric in metrics_to_plot:
            units = UNITS.get(metric, '')
            label = f"{metric}\n({units})" if units else metric
            labels_with_units.append(label)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(labels_with_units, rotation=45, ha="right")
        ax.set_ylabel('Values')
        ax.legend()

    fig_path = os.path.join('plots', f'{model_name}_example_comparisons.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved example comparison plot to {fig_path}")