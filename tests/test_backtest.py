"""Tests for corrected pair-trading accounting."""
import numpy as np
import pandas as pd

from src.backtest import backtest_pair_strategy, pair_weights, performance_metrics


def test_pair_weights_are_gross_normalized():
    idx = pd.date_range("2020-01-01", periods=2)
    positions = pd.Series([1.0, -1.0], index=idx)
    beta = pd.Series([0.5, 2.0], index=idx)

    weights = pair_weights(positions, beta)

    gross = weights["gold_weight"].abs() + weights["silver_weight"].abs()
    assert np.allclose(gross, 1.0)
    assert np.allclose(weights["gross_exposure"], [1.5, 3.0])


def test_backtest_charges_leg_turnover_costs():
    idx = pd.date_range("2020-01-01", periods=3)
    gold = pd.Series([100.0, 100.0, 100.0], index=idx)
    silver = pd.Series([50.0, 50.0, 50.0], index=idx)
    beta = pd.Series([1.0, 1.0, 1.0], index=idx)
    positions = pd.Series([0.0, 1.0, 1.0], index=idx)

    result = backtest_pair_strategy(positions, gold, silver, beta, cost_bps=10.0)

    assert result.pnl_gross.sum() == 0.0
    assert result.costs.iloc[1] == 0.001
    assert result.pnl_net.sum() == -0.001
    assert result.equity.iloc[-1] < 1.0


def test_backtest_normalization_reduces_return_magnitude():
    idx = pd.date_range("2020-01-01", periods=3)
    gold = pd.Series([100.0, 110.0, 110.0], index=idx)
    silver = pd.Series([50.0, 50.0, 50.0], index=idx)
    beta = pd.Series([1.0, 1.0, 1.0], index=idx)
    positions = pd.Series([1.0, 1.0, 1.0], index=idx)

    normalized = backtest_pair_strategy(
        positions, gold, silver, beta, cost_bps=0.0, normalize_gross=True
    )
    raw = backtest_pair_strategy(
        positions, gold, silver, beta, cost_bps=0.0, normalize_gross=False
    )

    assert np.isclose(raw.pnl_net.iloc[1], 0.10)
    assert np.isclose(normalized.pnl_net.iloc[1], 0.05)


def test_performance_metrics_are_finite_for_basic_series():
    idx = pd.date_range("2020-01-01", periods=4)
    pnl = pd.Series([0.0, 0.01, -0.005, 0.002], index=idx)
    equity = (1 + pnl).cumprod()

    metrics = performance_metrics(pnl, equity, name="demo")

    assert metrics["Strategy"] == "demo"
    assert metrics["N"] == 4
    assert np.isfinite(metrics["Sharpe"])
