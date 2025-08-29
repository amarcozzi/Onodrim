"""
evaluation.py
"""
import torch
import numpy as np
import pandas as pd

from visualization import (
    visualize_embeddings,
    visualize_metric_performance,
    display_similarity_examples,
    calculate_regression_metrics,
    calculate_classification_metrics,
    UNITS
)

MIN_DISTANCES = []

def find_most_similar_plots(model, scaler, all_data, feature_cols, plot_id_col, test_features, test_ids, top_k=5):
    """Finds the most similar plots for each test sample."""
    model.eval()
    X_all = all_data.select(feature_cols).to_numpy()
    X_all = np.nan_to_num(X_all, nan=0.0)
    X_all_scaled = scaler.transform(X_all)
    plot_ids_all = all_data.select(plot_id_col).to_numpy().flatten()
    X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)

    with torch.no_grad():
        all_embeddings = model.get_embedding(X_all_tensor).numpy()
        test_tensor = torch.tensor(test_features, dtype=torch.float32)
        test_embeddings = model.get_embedding(test_tensor).numpy()

    results = []
    for i, (test_embedding, true_plot_id) in enumerate(zip(test_embeddings, test_ids)):
        distances = np.linalg.norm(all_embeddings - test_embedding, axis=1)
        mask = plot_ids_all != true_plot_id
        valid_distances, valid_plot_ids = distances[mask], plot_ids_all[mask]

        if len(valid_distances) >= top_k:
            top_indices = np.argsort(valid_distances)[:top_k]
            results.append({
                'true_plot_id': true_plot_id,
                'top_k_plots': list(zip(valid_plot_ids[top_indices], valid_distances[top_indices])),
                'distances': np.sort(valid_distances)[:top_k]
            })

            MIN_DISTANCES.append(np.min(valid_distances))
    return results

def evaluate_predictions(predictions_df, key_metrics_con, key_metrics_cat):
    """
    Calculates comprehensive performance metrics for continuous and categorical variables.
    """
    metrics = {}
    print("\n--- Quantitative Evaluation ---")

    # Continuous metrics
    for metric in key_metrics_con:
        true_vals = predictions_df[f'true_{metric}']
        pred_vals = predictions_df[f'pred_{metric}']
        rmse, mae, r2 = calculate_regression_metrics(true_vals, pred_vals)

        metrics[f'{metric}_rmse'] = rmse
        metrics[f'{metric}_mae'] = mae
        metrics[f'{metric}_r2'] = r2

        units = UNITS.get(metric, '')
        unit_str = f" {units}" if units else ""

        print(f"Metrics for {metric}:")
        print(f"  RMSE: {rmse:.4f}{unit_str}")
        print(f"  MAE: {mae:.4f}{unit_str}")
        print(f"  R²: {r2:.4f}")
        print()

    # Categorical metrics
    for metric in key_metrics_cat:
        true_vals = predictions_df[f'true_{metric}']
        pred_vals = predictions_df[f'pred_{metric}']
        classification_metrics = calculate_classification_metrics(true_vals, pred_vals)

        # Store all classification metrics
        for key, value in classification_metrics.items():
            if key != 'full_report':  # Don't store the full report in metrics dict
                metrics[f'{metric}_{key}'] = value

        print(f"Classification metrics for {metric}:")
        print(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        print(f"  Macro Precision: {classification_metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {classification_metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score: {classification_metrics['macro_f1']:.4f}")
        print()

    return metrics

def run_evaluation(model, scaler, plot_data, feature_cols, plot_id_col, X_test, y_test,
                   key_metrics_con, key_metrics_cat, model_name):
    """
    Runs the full evaluation and visualization pipeline.
    """
    print("\n--- Starting Full Evaluation Pipeline ---")

    # 1. Visualize Embeddings
    visualize_embeddings(model, scaler, plot_data, feature_cols,
                         metrics_to_color_con=key_metrics_con,
                         metrics_to_color_cat=key_metrics_cat,
                         model_name=model_name)

    # 2. Find Similar Plots
    results = find_most_similar_plots(
        model, scaler, plot_data, feature_cols, plot_id_col, X_test, y_test, top_k=1
    )

    # 3. Prepare a DataFrame for analysis
    plot_data_pd = plot_data.to_pandas().set_index(plot_id_col)
    predictions = []
    all_metrics = key_metrics_con + key_metrics_cat

    for res in results:
        true_id = res['true_plot_id']
        top_k_ids = [item[0] for item in res['top_k_plots']]
        if true_id in plot_data_pd.index and all(pid in plot_data_pd.index for pid in top_k_ids):
            pred_row = {'true_plot_id': true_id, 'predicted_plot_id': top_k_ids[0]}  # Keep first as reference
            for metric in all_metrics:
                true_val = plot_data_pd.loc[true_id, metric]

                if metric in key_metrics_con:
                    # For continuous variables, calculate mean across all top-k plots
                    pred_vals = [plot_data_pd.loc[pid, metric] for pid in top_k_ids]
                    pred_val = np.mean(pred_vals)
                    error = pred_val - true_val
                    pred_row[f'true_{metric}'] = true_val
                    pred_row[f'pred_{metric}'] = pred_val
                    pred_row[f'error_{metric}'] = error
                    pred_row[f'pct_error_{metric}'] = (error / true_val * 100) if true_val != 0 else 0
                else:
                    # For categorical variables, use mode (most common value)
                    pred_vals = [plot_data_pd.loc[pid, metric] for pid in top_k_ids]
                    pred_val = pd.Series(pred_vals).mode()[0]  # Get the most common value
                    pred_row[f'true_{metric}'] = true_val
                    pred_row[f'pred_{metric}'] = pred_val

        predictions.append(pred_row)

    predictions_df = pd.DataFrame(predictions)

    # 4. Quantitative Evaluation
    metrics = evaluate_predictions(predictions_df, key_metrics_con, key_metrics_cat)

    # 5. Visual Evaluation - store detailed metrics from visualizations
    detailed_metrics = {}

    for metric in key_metrics_con:
        viz_metrics = visualize_metric_performance(predictions_df, metric, is_categorical=False, model_name=model_name)
        detailed_metrics[f'{metric}_detailed'] = viz_metrics

    for metric in key_metrics_cat:
        viz_metrics = visualize_metric_performance(predictions_df, metric, is_categorical=True, model_name=model_name)
        detailed_metrics[f'{metric}_detailed'] = viz_metrics

    display_similarity_examples(results, plot_data, plot_id_col, all_metrics, model_name, num_examples=5)

    # Combine metrics
    all_metrics_combined = {**metrics, **detailed_metrics}

    return all_metrics_combined