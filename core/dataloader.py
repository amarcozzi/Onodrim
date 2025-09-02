"""
dataloader.py
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ForestDataset(Dataset):
    """Custom PyTorch Dataset for forest plot data."""
    def __init__(self, features, plot_ids):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.plot_ids = torch.tensor(plot_ids, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.plot_ids[idx]

def prepare_data(plot_data, feature_cols, plot_id_col, batch_size=64, test_size=0.2, random_state=42):
    """
    Prepares the data for training and testing.

    Args:
        plot_data (pl.DataFrame): The input data.
        feature_cols (list): A list of feature column names.
        plot_id_col (str): The name of the plot ID column.
        batch_size (int): The batch size for the DataLoader.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.

    Returns:
        tuple: A tuple containing train_loader, test_loader, scaler, X_test, and y_test.
    """
    X = plot_data.select(feature_cols).to_numpy()
    plot_ids = plot_data.select(plot_id_col).to_numpy().flatten()

    if np.isnan(X).any():
        print(f"Warning: Found {np.isnan(X).sum()} NaN values in features. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, plot_ids, test_size=test_size, random_state=random_state)

    train_dataset = ForestDataset(X_train, y_train)
    test_dataset = ForestDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, X_test, y_test

def create_output_directories():
    """Creates directories for saving plots and model weights if they don't exist."""
    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)
    print("Created 'plots' and 'weights' directories.")