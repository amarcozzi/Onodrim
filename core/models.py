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


class SelfAttention(nn.Module):
    """Self-attention layer for tabular data."""
    def __init__(self, input_dim, attention_dim=None):
        super(SelfAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // 2
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))

    def forward(self, x):
        x_reshaped = x.unsqueeze(1)
        q = self.query(x_reshaped)
        k = self.key(x_reshaped)
        v = self.value(x_reshaped)
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention, dim=-1)
        context = torch.matmul(attention_weights, v)
        output = context.squeeze(1)
        return output + x


class AttentionAutoencoder(nn.Module):
    """An autoencoder with residual blocks and self-attention."""
    def __init__(self, input_dim, latent_dim=64, hidden_dims=(128, 96), dropout_rate=0.2, use_attention=True):
        super(AttentionAutoencoder, self).__init__()
        encoder_layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            encoder_layers.append(ResidualBlock(prev_dim, dim))
            encoder_layers.append(nn.Dropout(dropout_rate))
            if use_attention:
                encoder_layers.append(SelfAttention(dim))
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(hidden_dims):
            decoder_layers.append(ResidualBlock(prev_dim, dim))
            decoder_layers.append(nn.Dropout(dropout_rate))
            if use_attention:
                decoder_layers.append(SelfAttention(dim))
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def get_embedding(self, x):
        return self.encoder(x)