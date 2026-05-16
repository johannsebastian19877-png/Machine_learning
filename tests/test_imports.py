"""Smoke tests: todas las librerias del pipeline deben importar."""
import pytest


def test_core_imports():
    import numpy, pandas, scipy, matplotlib  # noqa: F401


def test_yfinance_import():
    import yfinance  # noqa: F401


def test_statsmodels_import():
    import statsmodels.api  # noqa: F401
    from statsmodels.tsa.stattools import adfuller  # noqa: F401
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: F401


def test_pykalman_import():
    from pykalman import KalmanFilter  # noqa: F401


def test_arch_import():
    from arch import arch_model  # noqa: F401


def test_hmmlearn_import():
    from hmmlearn.hmm import GaussianHMM  # noqa: F401


def test_torch_import():
    import torch  # noqa: F401
    assert torch.tensor([1.0]).item() == 1.0


def test_gymnasium_import():
    import gymnasium  # noqa: F401


def test_stable_baselines3_import():
    from stable_baselines3 import PPO  # noqa: F401


def test_sklearn_import():
    import sklearn  # noqa: F401
    from sklearn.metrics import roc_auc_score  # noqa: F401
