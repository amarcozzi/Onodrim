"""
loss.py
"""
import torch
import torch.nn as nn

class CustomMSELoss(nn.Module):
    """Custom Mean Squared Error loss with feature weighting."""
    def __init__(self, weights=None):
        super(CustomMSELoss, self).__init__()
        self.weights = weights

    def forward(self, inputs, targets):
        sq_error = (inputs - targets) ** 2
        if self.weights is not None:
            sq_error = sq_error * self.weights
        return torch.mean(sq_error)