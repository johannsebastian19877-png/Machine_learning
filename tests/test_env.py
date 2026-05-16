"""Tests del entorno custom Gymnasium."""
import numpy as np
import pytest

from envs.pair_trading_env import PairTradingEnv


def test_env_creation(synthetic_panel):
    env = PairTradingEnv(synthetic_panel, cost_bps=5.0, random_start=False, seed=0)
    assert env.observation_space.shape == (len(PairTradingEnv.OBS_COLS) + 1,)
    assert env.action_space.n == 3


def test_env_reset(synthetic_panel):
    env = PairTradingEnv(synthetic_panel, random_start=False, seed=0)
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.shape == (len(PairTradingEnv.OBS_COLS) + 1,)
    assert env.position == 0
    assert env.equity == 1.0


def test_env_step_returns_5tuple(synthetic_panel):
    env = PairTradingEnv(synthetic_panel, random_start=False, seed=0)
    env.reset()
    obs, r, term, trunc, info = env.step(1)
    assert isinstance(obs, np.ndarray)
    assert isinstance(r, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "equity" in info and "position" in info


def test_env_flat_yields_zero_pnl(synthetic_panel):
    """Politica always-flat NO debe tener variacion de equity ni costos."""
    env = PairTradingEnv(synthetic_panel, cost_bps=5.0,
                         episode_len=len(synthetic_panel) - 2,
                         random_start=False, seed=0)
    env.reset()
    rewards = []
    while True:
        _, r, term, trunc, info = env.step(1)  # accion 1 = flat
        rewards.append(r)
        if term or trunc: break
    assert all(r == 0 for r in rewards), "Flat policy debe dar reward 0"
    assert env.equity == 1.0, "Equity debe permanecer en 1.0"
    assert env.trades == 0, "Flat policy no debe abrir trades"


def test_env_determinism(synthetic_panel):
    """Mismas acciones → mismos resultados."""
    actions = [2, 2, 0, 0, 1, 2, 1] * 30

    def run():
        env = PairTradingEnv(synthetic_panel, random_start=False, seed=42)
        env.reset()
        eqs = []
        for a in actions:
            _, _, _, trunc, info = env.step(a)
            eqs.append(info["equity"])
            if trunc: break
        return eqs

    r1, r2 = run(), run()
    assert r1 == r2, "Env debe ser deterministico"


def test_env_costs_applied(synthetic_panel):
    """Cambiar de posicion repetidamente debe erosionar el equity por costos."""
    env = PairTradingEnv(synthetic_panel, cost_bps=50.0,  # costo alto para visualizar
                         episode_len=200, random_start=False, seed=0)
    env.reset()
    actions = [2, 0] * 100  # alterna long/short cada dia
    for a in actions:
        _, _, term, trunc, _ = env.step(a)
        if term or trunc: break
    assert env.equity < 1.0, "Costos altos + churn debe destruir equity"
    assert env.trades > 0


def test_env_missing_column_raises(synthetic_panel):
    df_bad = synthetic_panel.drop(columns=["spread_z"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        PairTradingEnv(df_bad)


def test_env_action_mapping(synthetic_panel):
    env = PairTradingEnv(synthetic_panel, random_start=False, seed=0)
    env.reset()
    env.step(0)  # short
    assert env.position == -1
    env.step(1)  # flat
    assert env.position == 0
    env.step(2)  # long
    assert env.position == 1


def test_env_truncation(synthetic_panel):
    """El episodio termina con truncated=True al alcanzar episode_len."""
    env = PairTradingEnv(synthetic_panel, episode_len=10,
                         random_start=False, seed=0)
    env.reset()
    truncated = False
    for _ in range(20):
        _, _, term, trunc, _ = env.step(1)
        if trunc:
            truncated = True
            break
    assert truncated, "Episodio debe truncarse al alcanzar episode_len"
