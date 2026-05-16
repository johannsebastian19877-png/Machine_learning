#!/usr/bin/env bash
# Ejecuta el pipeline completo (Linux/Mac)
# Uso:  bash run_all.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
NBS=(
  "01_data_ingestion.ipynb"
  "02_cointegration.ipynb"
  "03_dynamic_hedge.ipynb"
  "04_features.ipynb"
  "05_signals.ipynb"
  "06_deep_model.ipynb"
  "07_env_gym.ipynb"
  "08_rl_agent.ipynb"
  "09_backtest.ipynb"
)

echo "[INFO] Python: $($PYTHON --version)"
echo "[INFO] Pipeline de $((${#NBS[@]})) notebooks comenzando..."

for nb in "${NBS[@]}"; do
  echo ""
  echo "================================================================="
  echo "  Ejecutando $nb"
  echo "================================================================="
  $PYTHON -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "$nb"
done

echo ""
echo "[OK] Pipeline completo. Reporte final: data/final_metrics.csv"
