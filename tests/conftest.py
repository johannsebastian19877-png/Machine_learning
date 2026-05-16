"""Fixtures comunes para los tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Permitir importar el modulo envs desde la raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_panel():
    """Genera un panel sintetico con todas las columnas que el env espera.
    No requiere descarga de datos reales — ideal para CI."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.bdate_range("2020-01-01", periods=n, name="Date")

    # Random walks correlacionados para precios
    eps = rng.standard_normal((n, 2)) * 0.01
    log_g = np.cumsum(eps[:, 0]) + np.log(1800)
    log_s = np.cumsum(0.7 * eps[:, 0] + 0.7 * eps[:, 1]) + np.log(25)
    gold = np.exp(log_g)
    silver = np.exp(log_s)

    df = pd.DataFrame({
        "Gold": gold,
        "Silver": silver,
        "spread_z": rng.standard_normal(n),
        "beta_kalman": 0.75 + 0.05 * rng.standard_normal(n).cumsum() / np.sqrt(n),
        "garch_vol_gold": np.abs(rng.standard_normal(n)) * 0.01 + 0.01,
        "garch_vol_silver": np.abs(rng.standard_normal(n)) * 0.02 + 0.02,
        "hmm_state": rng.integers(0, 3, n).astype(float),
        "spread_ema_5": rng.standard_normal(n),
        "spread_ema_20": rng.standard_normal(n),
        "rsi_spread_14": rng.uniform(20, 80, n),
        "bollinger_pct_b": rng.uniform(0, 1, n),
        "gsr_z_60": rng.standard_normal(n),
        "lstm_prob_up": rng.uniform(0.3, 0.7, n),
    }, index=idx)
    return df


@pytest.fixture(scope="session")
def project_root():
    return ROOT
