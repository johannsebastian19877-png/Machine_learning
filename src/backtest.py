"""Backtesting and metrics helpers for a beta-hedged pair trade."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    pnl_gross: pd.Series
    costs: pd.Series
    pnl_net: pd.Series
    equity: pd.Series
    turnover: pd.Series
    gold_weight: pd.Series
    silver_weight: pd.Series
    gross_exposure: pd.Series


def pair_weights(
    positions: pd.Series,
    beta: pd.Series,
    *,
    normalize_gross: bool = True,
) -> pd.DataFrame:
    """Convert spread positions into gold/silver portfolio weights.

    ``position=+1`` means long spread: long gold, short ``beta`` units of
    silver. With gross normalization enabled, the absolute weights sum to one.
    """

    positions, beta = positions.align(beta, join="inner")
    beta_abs = beta.abs()
    gross_exposure = 1.0 + beta_abs
    denominator = gross_exposure if normalize_gross else 1.0

    gold_weight = positions / denominator
    silver_weight = -positions * beta / denominator

    return pd.DataFrame(
        {
            "gold_weight": gold_weight,
            "silver_weight": silver_weight,
            "gross_exposure": gross_exposure,
        }
    )


def backtest_pair_strategy(
    positions: pd.Series,
    gold_prices: pd.Series,
    silver_prices: pd.Series,
    beta: pd.Series,
    *,
    cost_bps: float = 5.0,
    normalize_gross: bool = True,
    charge_beta_rebalance: bool = True,
) -> BacktestResult:
    """Backtest a beta-hedged Gold/Silver strategy with realistic leg turnover.

    Signals are assumed to be known at the close of date ``t`` and earn returns
    from ``t`` to ``t+1``. In the returned daily PnL indexed by date ``t``, the
    effective weights are therefore the lagged target weights from ``t-1``.
    """

    data = pd.concat(
        {
            "position": positions.astype(float),
            "gold": gold_prices.astype(float),
            "silver": silver_prices.astype(float),
            "beta": beta.astype(float),
        },
        axis=1,
        join="inner",
    ).sort_index()

    beta_for_signal = data["beta"]
    target = pair_weights(
        data["position"],
        beta_for_signal,
        normalize_gross=normalize_gross,
    )

    effective_gold = target["gold_weight"].shift(1).fillna(0.0)
    effective_silver = target["silver_weight"].shift(1).fillna(0.0)
    effective_gross = target["gross_exposure"].shift(1).fillna(target["gross_exposure"])

    ret_gold = data["gold"].pct_change().fillna(0.0)
    ret_silver = data["silver"].pct_change().fillna(0.0)

    pnl_gross = effective_gold * ret_gold + effective_silver * ret_silver

    if charge_beta_rebalance:
        turnover = (
            target[["gold_weight", "silver_weight"]]
            .diff()
            .abs()
            .sum(axis=1)
            .fillna(target[["gold_weight", "silver_weight"]].abs().sum(axis=1))
        )
    else:
        turnover = data["position"].diff().abs().fillna(data["position"].abs()) * 2.0
        if normalize_gross:
            turnover = turnover / effective_gross.replace(0.0, np.nan).fillna(1.0)

    costs = turnover * (cost_bps / 10_000.0)
    pnl_net = (pnl_gross - costs).fillna(0.0)
    equity = (1.0 + pnl_net).cumprod()

    return BacktestResult(
        pnl_gross=pnl_gross.rename("pnl_gross"),
        costs=costs.rename("costs"),
        pnl_net=pnl_net.rename("pnl_net"),
        equity=equity.rename("equity"),
        turnover=turnover.rename("turnover"),
        gold_weight=effective_gold.rename("gold_weight"),
        silver_weight=effective_silver.rename("silver_weight"),
        gross_exposure=effective_gross.rename("gross_exposure"),
    )


def performance_metrics(
    pnl: pd.Series,
    equity: pd.Series,
    *,
    name: str,
    periods_per_year: int = 252,
) -> dict[str, float | int | str]:
    pnl = pnl.fillna(0.0)
    equity = equity.reindex(pnl.index).ffill()
    std = pnl.std()
    ann_ret = (1.0 + pnl).prod() ** (periods_per_year / len(pnl)) - 1.0 if len(pnl) else 0.0
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe = (pnl.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0.0
    downside = pnl[pnl < 0].std()
    sortino = (pnl.mean() / downside) * np.sqrt(periods_per_year) if downside > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_dd = drawdown.min() if len(drawdown) else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.inf
    active = pnl[pnl != 0]
    win_rate_active = (active > 0).mean() if len(active) else 0.0

    return {
        "Strategy": name,
        "N": int(len(pnl)),
        "AnnReturn": float(ann_ret),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDD": float(max_dd),
        "Calmar": float(calmar),
        "WinRate(active)": float(win_rate_active),
        "FinalEquity": float(equity.iloc[-1]) if len(equity) else 1.0,
    }

