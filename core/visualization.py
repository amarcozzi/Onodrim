"""
visualization.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

def visualize_embeddings(model, scaler, plot_data, feature_cols, color_by_col, save_path='embeddings_tsne.png'):
    """
    Visualizes the embeddings using t-SNE.

    Args:
        model (nn.Module): The trained model.
        scaler: The fitted StandardScaler.
        plot_data (pl.DataFrame): All plot data.
        feature_cols (list): List of feature column names.
        color_by_col (str): The column name to use for coloring the plot.
        save_path (str): The path to save the visualization.
    """
    model.eval()
    X = plot_data.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    color_values = plot_data.select(color_by_col).to_numpy().flatten()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        embeddings = model.get_embedding(X_tensor).numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=color_values, cmap='tab20', alpha=0.7, s=30)
    plt.colorbar(scatter, label=color_by_col)
    plt.title('t-SNE Visualization of Forest Plot Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Embedding visualization saved to {save_path}")

def display_similarity_results(results, plot_data, plot_id_col, key_metrics, num_examples=5):
    """
    Displays detailed information about the similarity search results.

    Args:
        results (list): The results from find_most_similar_plots.
        plot_data (pl.DataFrame): All plot data.
        plot_id_col (str): The name of the plot ID column.
        key_metrics (list): A list of metrics to display.
        num_examples (int): The number of random examples to show.
    """
    plot_data_pd = plot_data.to_pandas().set_index(plot_id_col)
    example_indices = random.sample(range(len(results)), min(num_examples, len(results)))

    for i, idx in enumerate(example_indices):
        result = results[idx]
        true_plot_id = result['true_plot_id']
        similar_plots = result['top_k_plots']

        print(f"\n--- Example {i + 1} ---")
        print(f"True Plot ID: {true_plot_id}")
        if true_plot_id in plot_data_pd.index:
            print("True Plot Metrics:")
            print(plot_data_pd.loc[true_plot_id, key_metrics])

        print("\nTop Similar Plots:")
        for j, (similar_plot_id, distance) in enumerate(similar_plots[:3]):
            print(f"{j + 1}. Plot ID: {similar_plot_id}, Distance: {distance:.4f}")
            if similar_plot_id in plot_data_pd.index:
                print(plot_data_pd.loc[similar_plot_id, key_metrics])