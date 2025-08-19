"""
evaluation.py
"""
import torch
import numpy as np
import pandas as pd

def find_most_similar_plots(model, scaler, all_data, feature_cols, plot_id_col, test_features, test_ids, top_k=5):
    """
    Finds the most similar plots for each test sample.

    Args:
        model (nn.Module): The trained autoencoder.
        scaler: The fitted StandardScaler.
        all_data (pl.DataFrame): All plot data.
        feature_cols (list): List of feature column names.
        plot_id_col (str): The name of the plot ID column.
        test_features (np.ndarray): The test features.
        test_ids (np.ndarray): The test plot IDs.
        top_k (int): The number of similar plots to find.

    Returns:
        list: A list of dictionaries with the similarity results.
    """
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
        distances = np.sqrt(np.sum((all_embeddings - test_embedding) ** 2, axis=1))
        different_plot_mask = plot_ids_all != true_plot_id
        valid_distances = distances[different_plot_mask]
        valid_plot_ids = plot_ids_all[different_plot_mask]

        if len(valid_distances) >= top_k:
            top_indices = np.argsort(valid_distances)[:top_k]
            top_plot_ids = valid_plot_ids[top_indices]
            top_distances = valid_distances[top_indices]
            results.append({
                'test_idx': i,
                'true_plot_id': true_plot_id,
                'top_k_plots': list(zip(top_plot_ids, top_distances))
            })
    return results

def evaluate_predictions(results, plot_data, plot_id_col, key_metrics):
    """
    Evaluates the similarity predictions by calculating RMSE for key metrics.

    Args:
        results (list): The results from find_most_similar_plots.
        plot_data (pl.DataFrame): All plot data.
        plot_id_col (str): The name of the plot ID column.
        key_metrics (list): A list of metrics to evaluate.

    Returns:
        dict: A dictionary of RMSE values for each metric.
    """
    plot_data_pd = plot_data.to_pandas().set_index(plot_id_col)
    errors_by_metric = {metric: [] for metric in key_metrics}

    for result in results:
        true_plot_id = result['true_plot_id']
        predicted_plot_id = result['top_k_plots'][0][0]

        if true_plot_id in plot_data_pd.index and predicted_plot_id in plot_data_pd.index:
            for metric in key_metrics:
                true_value = plot_data_pd.loc[true_plot_id, metric]
                pred_value = plot_data_pd.loc[predicted_plot_id, metric]
                error = (true_value - pred_value) ** 2
                errors_by_metric[metric].append(error)

    rmse_by_metric = {}
    for metric, errors in errors_by_metric.items():
        if errors:
            rmse = np.sqrt(np.mean(errors))
            rmse_by_metric[metric] = rmse
            print(f"RMSE for {metric}: {rmse:.4f}")

    return rmse_by_metric