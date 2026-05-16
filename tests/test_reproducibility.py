"""Tests de reproducibilidad — seeds y determinismo."""
import numpy as np
import torch


def test_numpy_seed():
    np.random.seed(42)
    a = np.random.randn(5)
    np.random.seed(42)
    b = np.random.randn(5)
    assert np.allclose(a, b)


def test_torch_seed():
    torch.manual_seed(42)
    a = torch.randn(5)
    torch.manual_seed(42)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_lstm_forward_deterministic():
    """Mismo modelo, misma entrada y mismo seed → misma salida."""
    import torch.nn as nn

    torch.manual_seed(42)
    m1 = nn.LSTM(input_size=5, hidden_size=8, batch_first=True)
    x = torch.zeros(2, 10, 5)
    m1.eval()
    with torch.no_grad():
        y1, _ = m1(x)

    torch.manual_seed(42)
    m2 = nn.LSTM(input_size=5, hidden_size=8, batch_first=True)
    m2.eval()
    with torch.no_grad():
        y2, _ = m2(x)

    assert torch.allclose(y1, y2)
