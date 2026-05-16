"""Verifica que los artifacts canonicos del proyecto estan presentes."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_NOTEBOOKS = [
    "01_data_ingestion.ipynb",
    "02_cointegration.ipynb",
    "03_dynamic_hedge.ipynb",
    "04_features.ipynb",
    "05_signals.ipynb",
    "06_deep_model.ipynb",
    "07_env_gym.ipynb",
    "08_rl_agent.ipynb",
    "09_backtest.ipynb",
]

EXPECTED_DOCS = ["README.md", "PLAN_PROYECTO.md", "AUDITORIA.md", "LICENSE"]
EXPECTED_CONFIG = ["requirements.txt", ".gitignore", "params.yaml", "Makefile"]


@pytest.mark.parametrize("nb", EXPECTED_NOTEBOOKS)
def test_notebook_exists(nb):
    assert (ROOT / nb).is_file(), f"Falta notebook {nb}"


@pytest.mark.parametrize("doc", EXPECTED_DOCS)
def test_doc_exists(doc):
    assert (ROOT / doc).is_file(), f"Falta documento {doc}"


@pytest.mark.parametrize("cfg", EXPECTED_CONFIG)
def test_config_exists(cfg):
    assert (ROOT / cfg).is_file(), f"Falta config {cfg}"


def test_env_module_importable():
    from envs.pair_trading_env import PairTradingEnv  # noqa: F401


def test_notebooks_are_valid_json():
    """Cada notebook debe ser JSON valido (nbformat parseable)."""
    import nbformat
    for nb in EXPECTED_NOTEBOOKS:
        nbformat.read(ROOT / nb, as_version=4)
