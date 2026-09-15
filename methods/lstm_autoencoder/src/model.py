from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_size: int = 24,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def reconstruction_errors(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - reconstructed) ** 2, dim=(1, 2))


def reconstruction_point_errors(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Return one reconstruction MSE per timestamp, averaged across features."""
    return torch.mean((x - reconstructed) ** 2, dim=2)
