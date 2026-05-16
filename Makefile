# =========================================================================
# Makefile — orquestacion del pipeline
# Uso:
#   make setup       Instala dependencias en el venv actual
#   make test        Corre la suite de tests rapidos
#   make data        Ejecuta solo notebooks 01-04 (datos + features)
#   make train       Ejecuta solo notebooks 05-08 (benchmark + ML + RL)
#   make backtest    Ejecuta solo notebook 09 (backtest final)
#   make all         Ejecuta el pipeline completo
#   make clean       Borra artifacts (data/, models/) — REGENERABLE
#   make lint        Linter sobre codigo Python (no notebooks)
# =========================================================================

PYTHON ?= python
NB_EXEC := jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800

.PHONY: setup test data train backtest all clean lint help

help:
	@echo "Targets disponibles: setup, test, data, train, backtest, all, clean, lint"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo ""
	@echo "Si necesitas PyTorch CPU especifico ejecuta:"
	@echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"

setup-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(NB_EXEC) 01_data_ingestion.ipynb
	$(NB_EXEC) 02_cointegration.ipynb
	$(NB_EXEC) 03_dynamic_hedge.ipynb
	$(NB_EXEC) 04_features.ipynb

train:
	$(NB_EXEC) 05_signals.ipynb
	$(NB_EXEC) 06_deep_model.ipynb
	$(NB_EXEC) 07_env_gym.ipynb
	$(NB_EXEC) 08_rl_agent.ipynb

backtest:
	$(NB_EXEC) 09_backtest.ipynb

all: data train backtest
	@echo ""
	@echo "=================================================="
	@echo "Pipeline completo OK. Revisa data/final_metrics.csv"
	@echo "=================================================="

clean:
	rm -rf data/*.csv data/*.parquet data/*.png data/*.json data/*.txt
	rm -rf models/*.pt models/*.zip
	rm -rf __pycache__ envs/__pycache__ tests/__pycache__ .pytest_cache
	rm -rf logs/

lint:
	$(PYTHON) -m ruff check envs/ tests/
