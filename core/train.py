"""
train.py
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from loss import CustomMSELoss

def train_autoencoder(model, model_name, train_loader, test_loader, feature_cols, feature_weights=None,
                      learning_rate=0.001, num_epochs=150, patience=50):
    """
    Trains an autoencoder model.

    Args:
        model (nn.Module): The autoencoder model to train.
        model_name (str): The name of the model.
        train_loader (DataLoader): The DataLoader for the training set.
        test_loader (DataLoader): The DataLoader for the test set.
        feature_cols (list): List of feature column names.
        feature_weights (dict): A dictionary mapping feature names to weights.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for improvement before early stopping.

    Returns:
        tuple: A tuple containing the trained model, training losses, and validation losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    if feature_weights:
        weights_tensor = torch.tensor([feature_weights.get(col, 1.0) for col in feature_cols], dtype=torch.float32)
        criterion = CustomMSELoss(weights_tensor)
    else:
        criterion = CustomMSELoss()

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, _ in train_loader:
            reconstructed, _ = model(features)
            loss = criterion(reconstructed, features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, _ in test_loader:
                reconstructed, _ = model(features)
                loss = criterion(reconstructed, features)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'Training complete. Best validation loss: {best_loss:.4f}')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.savefig(f'plots/{model_name}_training_curve.png')
    plt.close()

    return model, train_losses, val_losses