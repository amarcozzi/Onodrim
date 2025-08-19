"""
evaluation.py
"""
import torch
import numpy as np
import pandas as pd

from visualization import (
    visualize_embeddings,
    visualize_metric_performance,
    display_similarity_examples
)

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
                'top_k_plots': list(zip(valid_plot_ids[top_indices], valid_distances[top_indices]))
            })
    return results

def evaluate_predictions(predictions_df, key_metrics_con, key_metrics_cat):
    """
    Calculates performance metrics for continuous (RMSE) and categorical (Accuracy) variables.
    """
    metrics = {}
    print("\n--- Quantitative Evaluation ---")
    for metric in key_metrics_con:
        rmse = np.sqrt(np.mean(predictions_df[f'error_{metric}'] ** 2))
        metrics[f'{metric}_rmse'] = rmse
        print(f"RMSE for {metric}: {rmse:.4f}")

    for metric in key_metrics_cat:
        accuracy = (predictions_df[f'true_{metric}'] == predictions_df[f'pred_{metric}']).mean()
        metrics[f'{metric}_accuracy'] = accuracy
        print(f"Accuracy for {metric}: {accuracy:.2%}")

    return metrics

def run_evaluation(model, scaler, plot_data, feature_cols, plot_id_col, X_test, y_test,
                   key_metrics_con, key_metrics_cat, model_name):
    """
    Runs the full evaluation and visualization pipeline.
    """
    print("\n--- Starting Full Evaluation Pipeline ---")

    # 1. Visualize Embeddings
    visualize_embeddings(model, scaler, plot_data, feature_cols,
                         metrics_to_color=key_metrics_con + key_metrics_cat,
                         model_name=model_name)

    # 2. Find Similar Plots
    results = find_most_similar_plots(
        model, scaler, plot_data, feature_cols, plot_id_col, X_test, y_test, top_k=5
    )

    # 3. Prepare a DataFrame for analysis
    plot_data_pd = plot_data.to_pandas().set_index(plot_id_col)
    predictions = []
    all_metrics = key_metrics_con + key_metrics_cat

    for res in results:
        true_id = res['true_plot_id']
        pred_id = res['top_k_plots'][0][0]
        if true_id in plot_data_pd.index and pred_id in plot_data_pd.index:
            pred_row = {'true_plot_id': true_id, 'predicted_plot_id': pred_id}
            for metric in all_metrics:
                true_val = plot_data_pd.loc[true_id, metric]
                pred_val = plot_data_pd.loc[pred_id, metric]
                pred_row[f'true_{metric}'] = true_val
                pred_row[f'pred_{metric}'] = pred_val
                if metric in key_metrics_con:
                    error = pred_val - true_val
                    pred_row[f'error_{metric}'] = error
                    pred_row[f'pct_error_{metric}'] = (error / true_val * 100) if true_val != 0 else 0
            predictions.append(pred_row)

    predictions_df = pd.DataFrame(predictions)

    # 4. Quantitative Evaluation
    metrics = evaluate_predictions(predictions_df, key_metrics_con, key_metrics_cat)

    # 5. Visual Evaluation
    for metric in key_metrics_con:
        visualize_metric_performance(predictions_df, metric, is_categorical=False, model_name=model_name)
    for metric in key_metrics_cat:
        visualize_metric_performance(predictions_df, metric, is_categorical=True, model_name=model_name)

    display_similarity_examples(results, plot_data, plot_id_col, all_metrics, model_name, num_examples=5)

    return metrics