from __future__ import annotations

import torch
from torch import nn


class TemporalAttention(nn.Module):
    """Learn a normalized weight for each encoder timestamp (O(T) memory)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: [batch, time, hidden]; weights: [batch, time].
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return context, weights


class LSTMAutoencoder(nn.Module):
    """LSTM autoencoder with temporal attention and nonlinear latent projections.

    Input/output: [batch, time, input_size]. Existing constructor arguments
    and forward(x) remain compatible. Architecture changes require retraining;
    checkpoints from the original architecture are not directly compatible.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        latent_size: int = 24,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_size", input_size), ("hidden_size", hidden_size),
            ("latent_size", latent_size), ("num_layers", num_layers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout,
        )
        self.attention = TemporalAttention(hidden_size)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.latent_activation = nn.GELU()
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder_activation = nn.GELU()
        self.decoder = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=recurrent_dropout,
        )
        # Unbounded output supports negative standardized feature values.
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(2) != self.input_size:
            raise ValueError(
                f"x must have shape [batch, time, {self.input_size}], "
                f"got {tuple(x.shape)}"
            )
        if x.size(1) == 0:
            raise ValueError("x must contain at least one timestamp")
        encoded, _ = self.encoder(x)
        context, _ = self.attention(encoded)
        latent = self.latent_activation(self.to_latent(context))
        decoder_input = self.decoder_activation(self.from_latent(latent))
        repeated = decoder_input.unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def reconstruction_errors(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Return one reconstruction MSE per window, averaged over time/features."""
    return torch.mean((x - reconstructed) ** 2, dim=(1, 2))


def reconstruction_point_errors(x: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Return one reconstruction MSE per timestamp, averaged across features."""
    return torch.mean((x - reconstructed) ** 2, dim=2)

