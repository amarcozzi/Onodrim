"""
models.py
"""
import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    """A simple autoencoder with linear layers."""
    def __init__(self, input_dim, embedding_dim=32, dropout_rate=0.2):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)


class ResidualBlock(nn.Module):
    """A residual block with two linear layers."""
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.block(x)
        out += residual
        return self.relu(out)


class ResidualAutoencoder(nn.Module):
    """An autoencoder with residual blocks."""
    def __init__(self, input_dim, latent_dim=64, hidden_dims=(128, 96), dropout_rate=0.2):
        super(ResidualAutoencoder, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(ResidualBlock(prev_dim, dim))
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(ResidualBlock(prev_dim, dim))
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)


class GatedAttention(nn.Module):
    """Feature-wise self-attention for tabular data."""

    def __init__(self, input_dim, attention_dim=None):
        super(GatedAttention, self).__init__()
        if attention_dim is None:
            attention_dim = max(input_dim // 4, 8)

        # Generate attention weights for each feature
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(attention_dim, input_dim),
            nn.Sigmoid()  # Use sigmoid instead of softmax to allow multiple features to be important
        )

        # Value transformation
        self.value_transform = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # Compute attention weights for each feature
        attention_weights = self.attention_net(x)  # (batch_size, input_dim)

        # Transform values
        values = self.value_transform(x)  # (batch_size, input_dim)

        # Apply attention (element-wise multiplication)
        attended_features = values * attention_weights

        return attended_features + x  # Residual connection


class FeatureAttention(nn.Module):
    """
    Self-attention layer for tabular data features, treating each feature as an element in a sequence.
    This is a classic Key-Query-Value implementation.
    """

    def __init__(self, num_features, attention_dim=None):
        super(FeatureAttention, self).__init__()
        self.num_features = num_features
        if attention_dim is None:
            attention_dim = max(num_features // 4, 1)

        # Each feature is an item in the sequence. The "embedding" of each feature
        # is its scalar value, so the input dimension to these linear layers is 1.
        self.query = nn.Linear(1, attention_dim)
        self.key = nn.Linear(1, attention_dim)
        self.value = nn.Linear(1, 1)
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))

    def forward(self, x):
        # Input x has shape: (batch_size, num_features)

        # Reshape to (batch_size, num_features, 1) to treat features as a sequence
        # where each feature has an "embedding" of size 1.
        x_reshaped = x.unsqueeze(-1)

        # Q, K have shape (batch_size, num_features, attention_dim)
        q = self.query(x_reshaped)
        k = self.key(x_reshaped)

        # V has shape (batch_size, num_features, 1)
        v = self.value(x_reshaped)

        # Attention scores will have shape (batch_size, num_features, num_features)
        # This is the matrix of feature-to-feature relationships.
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale.to(x.device)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Context vector will have shape (batch_size, num_features, 1)
        context = torch.matmul(attention_weights, v)

        # Reshape back to (batch_size, num_features) and add residual connection
        output = context.squeeze(-1)
        return output + x


class PyTorchMultiheadAttention(nn.Module):
    """
    A wrapper for torch.nn.MultiheadAttention to make it compatible with tabular data
    in the shape (batch_size, num_features).
    """

    def __init__(self, num_features, num_heads=1):
        super(PyTorchMultiheadAttention, self).__init__()
        # For tabular data, we treat each feature as a token in a sequence.
        # The simplest embedding for a single feature's value is a dimension of 1.
        self.embed_dim = 1
        self.num_features = num_features
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=False)

    def forward(self, x):
        # Input x shape: (batch_size, num_features)

        # We need to reshape the input to what MultiheadAttention expects: (seq_len, batch_size, embed_dim).
        # Here, seq_len is num_features and embed_dim is 1.
        # 1. Add a dimension for the embedding: (batch_size, num_features, 1)
        x_reshaped = x.unsqueeze(-1)
        # 2. Permute to (num_features, batch_size, 1)
        x_permuted = x_reshaped.permute(1, 0, 2)

        # Apply attention
        attn_output, _ = self.multihead_attn(x_permuted, x_permuted, x_permuted)

        # Reshape back to the original format
        # 1. Permute back to (batch_size, num_features, 1)
        output_permuted = attn_output.permute(1, 0, 2)
        # 2. Squeeze the last dimension: (batch_size, num_features)
        output = output_permuted.squeeze(-1)

        # Add residual connection
        return output + x


class AttentionAutoencoder(nn.Module):
    """An autoencoder with attention followed by residual blocks."""

    def __init__(self, input_dim, latent_dim=64, hidden_dims=(128, 96), dropout_rate=0.2, attention_module=None):
        super(AttentionAutoencoder, self).__init__()

        # --- ENCODER ---
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            # 1. Apply attention to the current feature set
            if attention_module:
                # The attention layer maintains the dimension (prev_dim -> prev_dim)
                encoder_layers.append(attention_module(prev_dim))

            # 2. Process the attended features with a Residual Block
            # The Residual Block handles the dimension change (prev_dim -> dim)
            encoder_layers.append(ResidualBlock(prev_dim, dim))
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # --- DECODER ---
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            # 1. The Residual Block handles the dimension change (prev_dim -> dim)
            decoder_layers.append(ResidualBlock(prev_dim, dim))

            # 2. Apply attention to the processed features
            if attention_module:
                # The attention layer maintains the dimension (dim -> dim)
                decoder_layers.append(attention_module(dim))

            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)