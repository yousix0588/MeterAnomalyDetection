"""Real neural checks run only when a compatible PyTorch is installed."""
from pathlib import Path
import sys

import pytest

torch=pytest.importorskip("torch",reason="PyTorch is not installed; real neural runtime is not verified on this host")
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from methods.attention_lstm_autoencoder.src.model import LSTMAutoencoder


def test_attention_forward_shape_and_backward():
    model=LSTMAutoencoder(input_size=11,hidden_size=8,latent_size=4)
    x=torch.randn(2,288,11)
    y=model(x)
    assert y.shape==x.shape and torch.isfinite(y).all()
    ((x-y)**2).mean().backward()
    assert model.attention.score[0].weight.grad is not None


def test_original_weights_cannot_load_strictly():
    model=LSTMAutoencoder(input_size=11,hidden_size=8,latent_size=4)
    original={key:value for key,value in model.state_dict().items() if not key.startswith("attention.")}
    with pytest.raises(RuntimeError,match="attention"):
        model.load_state_dict(original,strict=True)
