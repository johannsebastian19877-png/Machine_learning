"""Custom Gym environment for pair-trading the Kalman spread Gold/Silver.

Estado (observacion): vector con [spread_z, beta_t, vol_gold, vol_silver, hmm_state,
spread_ema5, spread_ema20, rsi, bollinger_b, gsr_z, lstm_prob_up, position_actual].

Acciones: discretas {-1: short spread, 0: flat, 1: long spread}.

Reward: PnL neto del spread con pesos normalizados por exposicion bruta
(`1 + abs(beta)`) y costos proporcionales al turnover real de las dos piernas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.backtest import pair_weights


class PairTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    OBS_COLS = [
        "spread_z", "beta_kalman", "garch_vol_gold", "garch_vol_silver",
        "hmm_state", "spread_ema_5", "spread_ema_20", "rsi_spread_14",
        "bollinger_pct_b", "gsr_z_60", "lstm_prob_up",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        cost_bps: float = 5.0,
        turnover_penalty: float = 0.0,
        episode_len: int | None = None,
        random_start: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        missing = [c for c in self.OBS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")
        for needed in ("Gold", "Silver"):
            if needed not in df.columns:
                raise ValueError(f"Falta columna {needed} en df")

        self.df = df.copy().reset_index(drop=False)
        self.n = len(self.df)
        self.cost = cost_bps / 10_000.0
        self.turnover_penalty = turnover_penalty
        self.episode_len = episode_len if episode_len else self.n - 2
        self.random_start = random_start

        # +1 columna extra: posicion actual
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.OBS_COLS) + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=short, 1=flat, 2=long
        self._action_to_pos = {0: -1, 1: 0, 2: 1}

        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ---- helpers ----
    def _obs(self):
        row = self.df.iloc[self.t]
        feats = row[self.OBS_COLS].to_numpy(dtype=np.float32)
        return np.append(feats, np.float32(self.position))

    def _pnl_step(self, new_position: int) -> float:
        """PnL para mover de self.position -> new_position al cierre de t,
        observado al cierre de t+1."""
        # PnL del spread aplicado de t a t+1 con beta disponible en t.
        p_g_t  = self.df.iloc[self.t]["Gold"]
        p_g_n  = self.df.iloc[self.t + 1]["Gold"]
        p_s_t  = self.df.iloc[self.t]["Silver"]
        p_s_n  = self.df.iloc[self.t + 1]["Silver"]
        beta   = self.df.iloc[self.t]["beta_kalman"]
        ret_g  = (p_g_n - p_g_t) / p_g_t
        ret_s  = (p_s_n - p_s_t) / p_s_t
        weights = pair_weights(
            pd.Series([float(new_position)]),
            pd.Series([float(beta)]),
            normalize_gross=True,
        ).iloc[0]
        target_gold_weight = float(weights["gold_weight"])
        target_silver_weight = float(weights["silver_weight"])
        gross_pnl = target_gold_weight * ret_g + target_silver_weight * ret_s

        # Costos sobre turnover real de ambas piernas, incluyendo rehedge por beta.
        turnover = abs(target_gold_weight - self.gold_weight) + abs(
            target_silver_weight - self.silver_weight
        )
        cost = turnover * self.cost

        # Penalizacion de turnover (opcional, shaping)
        turnover_pen = self.turnover_penalty * turnover

        self.gold_weight = target_gold_weight
        self.silver_weight = target_silver_weight

        return gross_pnl - cost - turnover_pen

    # ---- API ----
    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.random_start:
            max_start = max(self.n - self.episode_len - 1, 1)
            self.t = int(self.rng.integers(0, max_start))
        else:
            self.t = 0
        self.position = 0
        self.gold_weight = 0.0
        self.silver_weight = 0.0
        self.equity = 1.0
        self.trades = 0
        self.steps = 0
        self.start_t = self.t
        return self._obs(), {}

    def step(self, action: int):
        new_pos = self._action_to_pos[int(action)]
        reward = self._pnl_step(new_pos)
        if new_pos != self.position:
            self.trades += 1
        self.position = new_pos
        self.equity *= (1 + reward)
        self.t += 1
        self.steps += 1

        terminated = False
        truncated = (self.steps >= self.episode_len) or (self.t >= self.n - 1)
        info = {
            "equity": self.equity, "position": self.position,
            "trades": self.trades, "date": str(self.df.iloc[self.t]["Date"]),
        }
        return self._obs(), float(reward), terminated, truncated, info
