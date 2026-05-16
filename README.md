# Arbitraje Estadistico Oro/Plata con Cointegracion Dinamica + Deep RL

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen)](tests/)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-blue)](https://github.com/astral-sh/ruff)

Pipeline end-to-end de **pair trading** sobre el spread Oro-Plata. Combina:
- **Cointegracion dinamica** via Kalman Filter (β adaptativo).
- **Deep Learning** (LSTM) para prediccion direccional del spread.
- **Reinforcement Learning** (PPO) para politica de ejecucion con costos de transaccion.

> ⚠️ **Disclaimer**: proyecto academico. No es consejo financiero ni listo para trading real.

---

## Quickstart

```bash
# 1. Clonar
git clone <repo-url>
cd project_final

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows

# 3. Instalar dependencias
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU PyTorch
pip install -r requirements.txt

# 4. Validar instalacion
pytest tests/ -v

# 5. Ejecutar pipeline completo (~10-15 min)
bash run_all.sh                    # Linux/Mac
.\run_all.ps1                      # Windows PowerShell
# o equivalente con Make:
make all
```

Al terminar, el reporte final esta en `data/final_metrics.csv` y `data/final_report.json`.

---

## Conda (alternativa)

```bash
conda env create -f environment.yml
conda activate pairtrading
pytest tests/ -v
make all
```

---

## Estructura del proyecto

```
project_final/
├── 01_data_ingestion.ipynb      # Quant Data Engineer
├── 02_cointegration.ipynb       # Cointegracion estatica (diagnostico)
├── 03_dynamic_hedge.ipynb       # Kalman filter (β adaptativo)
├── 04_features.ipynb            # GARCH, HMM, indicadores tecnicos
├── 05_signals.ipynb             # Benchmark Z-score
├── 06_deep_model.ipynb          # LSTM predictivo
├── 07_env_gym.ipynb             # Entorno Gymnasium custom
├── 08_rl_agent.ipynb            # Agente PPO
├── 09_backtest.ipynb            # Backtest comparativo out-of-sample
│
├── envs/
│   └── pair_trading_env.py      # MDP custom (importado por 7-9)
│
├── tests/                       # 41 tests unitarios + smoke
│   ├── conftest.py
│   ├── test_imports.py
│   ├── test_env.py
│   ├── test_reproducibility.py
│   └── test_project_structure.py
│
├── data/                        # Artifacts (gitignored — regenerables)
├── models/                      # LSTM + PPO entrenados (gitignored)
│
├── .github/workflows/ci.yml     # CI: tests en Ubuntu/Mac/Win × Py 3.10/3.11/3.12
├── requirements.txt             # Dependencias pinned
├── requirements-dev.txt         # + testing/linting
├── environment.yml              # Alternativa conda
├── params.yaml                  # Config central (referencia)
├── Makefile                     # Orquestacion
├── run_all.sh / run_all.ps1     # Scripts cross-platform
├── pyproject.toml               # pytest + ruff
│
├── PLAN_PROYECTO.md             # Diseño detallado de las 9 fases
├── AUDITORIA.md                 # Informe de auditoria final
└── LICENSE                      # MIT
```

---

## Orden de ejecucion (los notebooks dependen de los anteriores)

```
01 → genera gold_silver_panel.csv         (panel limpio)
02 → lee panel → spread_static.csv        (cointegracion estatica)
03 → lee panel → kalman_spread.csv        (Kalman β_t)
04 → lee panel + kalman → features.parquet (15 features causales)
05 → lee features → signals_benchmark.csv  (benchmark Z-score)
06 → lee features → models/lstm_best.pt + features_with_lstm.parquet
07 → demuestra el entorno custom
08 → entrena PPO → models/ppo_agent.zip
09 → backtest comparativo → final_metrics.csv + final_report.json
```

---

## Resultados (test set 2022+)

| Estrategia | Sharpe | Sortino | Max DD | Calmar | Final Equity |
|---|---|---|---|---|---|
| Z-score Benchmark | **+0.13** | +0.09 | -13.2% | +0.06 | 1.032 |
| LSTM Policy | +0.05 | +0.06 | -31.2% | -0.01 | 0.984 |
| PPO Agent | -0.46 | -0.61 | -36.1% | -0.23 | 0.688 |

### Hallazgos clave

- **Cointegracion rescatada con Kalman**: 4.9% → **97.8%** de ventanas estacionarias.
- **LSTM**: AUC 0.66, accuracy 60.5% en test (sin overfitting).
- **PPO subentrenado** (100k timesteps); requeriria 1M+ para producion.
- Reporte completo en [AUDITORIA.md](AUDITORIA.md).

---

## Comandos utiles

```bash
make test         # Tests unitarios
make data         # Solo notebooks 01-04 (datos + features)
make train        # Notebooks 05-08 (benchmark + LSTM + PPO)
make backtest     # Notebook 09 (backtest final)
make all          # Pipeline completo
make clean        # Borrar artifacts (data/, models/)
make lint         # Ruff sobre envs/ y tests/
```

---

## Reproducibilidad

| Garantia | Mecanismo |
|---|---|
| **Versiones fijas** | `requirements.txt` con rangos compatibles |
| **Seeds deterministicos** | `torch.manual_seed(42)`, `np.random.seed(42)`, PPO `seed=42` |
| **Paths cross-platform** | `pathlib.Path` en todo el codigo |
| **CI multi-plataforma** | GitHub Actions: Ubuntu/Mac/Win × Py 3.10/3.11/3.12 |
| **Tests automaticos** | 41 tests cubren imports, env, determinismo, estructura |
| **Anti-leakage** | Standardizers `fit` solo en train; features causales |

**Nota sobre los datos**: `yfinance` descarga datos hasta el dia actual. Para reproducibilidad estricta, fijar `END_DATE` en `params.yaml` y propagar al notebook 01.

---

## Stack tecnico

| Capa | Libreria |
|---|---|
| Datos | yfinance, pandas, numpy, pyarrow |
| Econometria | statsmodels (ADF, Johansen, OLS), arch (GARCH), pykalman |
| ML clasico | scikit-learn, hmmlearn |
| Deep Learning | PyTorch |
| RL | gymnasium, stable-baselines3 |
| Visualizacion | matplotlib, seaborn |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## Para colaboradores

Ver [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citacion

Si usas este proyecto, cita como:

```
Proyecto ML — Arbitraje Estadistico Oro/Plata (GSR), 2026.
Universidad — Curso Machine Learning, Octavo semestre.
```

## Licencia

MIT — ver [LICENSE](LICENSE).
