# Plan de Proyecto — Arbitraje Estadistico Oro/Plata con Hedge Ratio Dinamico y Reinforcement Learning

> **Version refinada** del proyecto original "ML Arbitraje Estadistico Oro-Plata (GSR)".
> Incorpora hallazgos empiricos de la Fase 1-2 (no hay cointegracion global con β fijo) y propone una arquitectura robusta que combina econometria adaptativa, deep learning predictivo y aprendizaje por refuerzo para ejecucion.

---

## 1. Resumen Ejecutivo

### 1.1 Objetivo
Construir un sistema de **arbitraje estadistico market-neutral** entre Oro (GC=F) y Plata (SI=F) que:
1. Modele la relacion estructural entre ambos metales como un proceso **dinamico** (no estatico).
2. Prediga la trayectoria del spread con **modelos secuenciales** (LSTM/GRU).
3. Ejecute decisiones de trading via un **agente de Reinforcement Learning** que optimiza Sharpe ajustado por costos.

### 1.2 Hipotesis central
> La relacion de equilibrio entre Oro y Plata existe, pero su parametro de equilibrio (β) **varia en el tiempo** debido a cambios estructurales (demanda industrial de plata, politica monetaria, regimenes de inflacion). Por tanto, la cointegracion debe modelarse con un **hedge ratio adaptativo**, no constante.

### 1.3 Metricas de exito (KPIs)
| KPI | Objetivo | Justificacion |
|---|---|---|
| **Sharpe Ratio (anualizado)** | ≥ 1.5 | Estandar de fondos sistematicos |
| **Sortino Ratio** | ≥ 2.0 | Penaliza solo downside |
| **Max Drawdown** | ≤ 15% | Limite de tolerancia al riesgo |
| **Calmar Ratio** | ≥ 1.0 | Retorno anual / MaxDD |
| **Beta vs S&P500** | |β| ≤ 0.2 | Validar market-neutrality |
| **% trades ganadores** | ≥ 55% | Solidez estadistica de la señal |
| **Spread estacionario** (ADF p) | < 0.05 en ventana | Validacion econometrica continua |

---

## 2. Diagnostico Empirico — Lo que ya sabemos

Resultados obtenidos en Fases 1 y 2 (ver [01_data_ingestion.py](01_data_ingestion.py) y [02_cointegration.py](02_cointegration.py)):

### 2.1 Datos validados ✅
- **4,116 dias** de cotizacion comun (2010-01-01 → 2026-05-16).
- **Vol anualizada**: Oro 17.04%, Plata 34.10% → ratio 2.0× (coherente).
- **Correlacion log-retornos**: 0.79 (rango esperado 0.6-0.85).

### 2.2 Cointegracion global RECHAZADA ⚠️
| Test | Estadistico | p-valor / Critico | Decision |
|---|---|---|---|
| **OLS** β = 0.7595, R² = 0.69, **DW = 0.003** | — | — | Red flag: regresion espuria |
| **ADF (Engle-Granger)** | -1.724 | p = 0.419 | No rechaza H0 (no estacionario) |
| **Johansen Trace** (r=0) | 7.08 | crit 95% = 15.49 | r = 0 (sin vector de cointegracion) |
| **Johansen Max-Eigen** (r=0) | 7.07 | crit 95% = 14.26 | r = 0 (sin cointegracion) |

### 2.3 Implicacion
El supuesto inicial del proyecto (β constante implica spread estacionario) **no se sostiene empiricamente** sobre la muestra completa. La relacion Oro-Plata se rompio estructuralmente al menos una vez en el periodo. **Solucion**: hedge ratio dinamico via Kalman Filter + verificacion de cointegracion en ventana movil.

---

## 3. Arquitectura del Sistema

```
┌────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE TRADING ALGORITMICO                  │
└────────────────────────────────────────────────────────────────────┘

  [yfinance API]
        │
        ▼
  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
  │  FASE 1      │    │  FASE 2          │    │  FASE 3              │
  │  Ingesta &   │───▶│  Cointegracion   │───▶│  Hedge Ratio         │
  │  Limpieza    │    │  Estatica (EG +  │    │  Dinamico (Kalman)   │
  │              │    │  Johansen)       │    │  + Cointegracion     │
  │              │    │  → Diagnostico   │    │  Rolling             │
  └──────────────┘    └──────────────────┘    └──────────────────────┘
                                                        │
                                                        ▼
  ┌──────────────────────┐    ┌──────────────────────────────────────┐
  │  FASE 4              │    │  FASE 5                              │
  │  Feature Engineering │◀───│  Spread Dinamico z_t = α + β_t·s_t   │
  │  (GARCH vol, RSI,    │    │  + Z-score Adaptativo                │
  │   EMA spread, GSR,   │    └──────────────────────────────────────┘
  │   regimen HMM)       │                          │
  └──────────────────────┘                          │
                │                                   │
                └──────────────┬────────────────────┘
                               ▼
                  ┌────────────────────────────┐
                  │  FASE 6                    │
                  │  Modelo Predictivo         │
                  │  LSTM / GRU + Walk-Forward │
                  │  → P(spread sube/baja)     │
                  └────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  FASE 7                    │
                  │  Entorno Gym Custom        │
                  │  (estados, acciones,       │
                  │   rewards con costos)      │
                  └────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  FASE 8                    │
                  │  Agente RL (PPO)           │
                  │  Politica de ejecucion     │
                  └────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  FASE 9                    │
                  │  Backtesting Out-of-Sample │
                  │  + Auditoria de KPIs       │
                  └────────────────────────────┘
```

---

## 4. Fases del Proyecto (Refinadas)

### FASE 1 — Ingesta de Datos ✅ COMPLETADA
**Entregable**: [01_data_ingestion.py](01_data_ingestion.py), `data/gold_silver_panel.csv`

- Descarga `GC=F` (Oro) y `SI=F` (Plata) via yfinance, ajustados, 2010-presente.
- Inner-join por fecha, ffill→bfill para NaN residuales.
- Calculo de retornos logaritmicos diarios.
- Validacion financiera (vol, correlacion, curtosis).
- Plot Base 100 normalizado.

---

### FASE 2 — Cointegracion Estatica (Diagnostico) ✅ COMPLETADA
**Entregable**: [02_cointegration.py](02_cointegration.py), `data/spread_bands.png`

- OLS para hedge ratio global β.
- ADF (Engle-Granger) sobre el spread.
- Johansen (Trace y Max-Eigenvalue) para robustez.
- **Resultado**: rechazo de cointegracion global → motiva Fase 3.

---

### FASE 3 — Hedge Ratio Dinamico (Kalman Filter) 🔜 SIGUIENTE
**Entregable**: `03_dynamic_hedge.py`, `data/kalman_spread.csv`

#### Justificacion
Un β estatico promedia toda la historia. Pero la "correa" entre Oro y Plata cambia: pre-2013 era ~50, post-2020 esta cerca de 80. Un Kalman Filter actualiza β_t **cada dia** con la nueva observacion, asumiendo:

```
β_t = β_{t-1} + w_t          (β evoluciona como random walk)
ln(P_oro,t) = β_t · ln(P_plata,t) + α + v_t
```

#### Tareas
1. Implementar Kalman Filter bivariado (paquete `pykalman` o manual).
2. Generar serie β_t y spread dinamico ε_t = ln(P_oro,t) − β_t · ln(P_plata,t).
3. **Cointegracion rolling**: ADF sobre ventana movil de 252 dias para identificar periodos cointegrados.
4. Comparar β_kalman vs β_ols_rolling(60/120/252).
5. Plot: β_t en el tiempo + spread dinamico con bandas adaptativas.

#### Criterio de exito
- Spread Kalman estacionario en **≥ 70% de las ventanas rolling** (ADF p < 0.05).
- β_t suave, sin saltos artificiales.

---

### FASE 4 — Feature Engineering
**Entregable**: `04_features.py`, `data/features.parquet`

#### Features a construir
| Categoria | Feature | Calculo |
|---|---|---|
| **Spread** | spread_z | (spread − μ_60) / σ_60 |
| **Spread** | spread_ema_5, ema_20 | Exponential MA |
| **Volatilidad** | garch_vol_gold, garch_vol_silver | GARCH(1,1) condicional |
| **Volatilidad** | spread_vol_realized | Std rolling 20d |
| **Tecnicos** | rsi_spread_14 | Relative Strength Index |
| **Tecnicos** | bollinger_pct_b | Posicion en bandas |
| **Ratio** | gsr = P_oro / P_plata | Gold-Silver Ratio |
| **Ratio** | gsr_z_60 | Z-score del GSR |
| **Regimen** | hmm_state | Hidden Markov Model 2-3 estados |
| **Momentum** | ret_5d_gold, ret_5d_silver | Retornos acumulados |
| **Cross-asset** | dxy_return, vix_level | Dolar Index, VIX (opcional) |

#### Justificacion del HMM
El HMM detecta automaticamente **regimenes** (alta vol, baja vol, ruptura estructural) sin etiquetas. Es la formalizacion estadistica del concepto "el mercado cambio".

---

### FASE 5 — Spread Dinamico y Señales Base
**Entregable**: `05_signals.py`, `data/signals.csv`

- Construir Z-score adaptativo del spread Kalman.
- Definir señales base (clasicas, sin ML aun) como **benchmark**:
  - `entry_long`: spread_z < −2.0
  - `entry_short`: spread_z > +2.0
  - `exit`: |spread_z| < 0.5
  - `stop`: |spread_z| > 4.0
- Calcular metricas del benchmark (Sharpe sin ML) → este es el numero a **superar** con ML.

---

### FASE 6 — Modelo Predictivo (LSTM/GRU)
**Entregable**: `06_deep_model.py`, `models/lstm_best.pt`

#### Arquitectura
```
Input: secuencia [t-60, t] de N features
   │
   ▼
LSTM (2 capas, hidden=64, dropout=0.2)
   │
   ▼
Dense (32, ReLU)
   │
   ▼
Output: 
   - clasificacion: P(spread sube en h dias) [softmax 3 clases: sube/lateral/baja]
   - regresion: E[Δspread en h dias]
```

#### Tareas
1. Construir tensores `(N, lookback=60, features=15)`.
2. Split temporal: train 2010-2019, val 2020-2022, test 2023-presente.
3. **Walk-forward validation**: reentrenar cada 6 meses.
4. Comparar LSTM vs GRU vs Transformer encoder pequeño (opcional).
5. Loss: CrossEntropy (clasificacion) o Huber (regresion).
6. Metricas: F1-score, Brier score, AUC-ROC.

#### Anti-overfitting
- Dropout, early stopping, label smoothing.
- Regularizacion L2 en pesos.
- No usar features de futuro (look-ahead bias check).

---

### FASE 7 — Entorno de Simulacion (Gym)
**Entregable**: `07_env_gym.py`, `envs/PairTradingEnv.py`

#### Definicion del MDP
| Componente | Definicion |
|---|---|
| **Estado s_t** | [spread_z, β_t, vol_garch, hmm_state, posicion_actual, prob_lstm, cash_ratio] |
| **Acciones** | {−1: short spread, 0: flat, +1: long spread} (discreto) o continuo en [-1,1] |
| **Reward r_t** | ΔPnL_t − λ·|Δposicion|·costo_tx − γ·vol_realizada |
| **Costos** | 5 bps por trade (comision + slippage realista) |
| **Episodio** | 1 año de trading (252 dias) |

#### Tareas
1. Implementar `gymnasium.Env` custom.
2. Manejo correcto de posiciones long/short con margen.
3. Tracking de equity curve, posiciones, trades.
4. Validar con politica trivial (siempre flat → reward = 0).

---

### FASE 8 — Agente de Reinforcement Learning
**Entregable**: `08_rl_agent.py`, `models/ppo_agent.zip`

#### Algoritmo
**PPO (Proximal Policy Optimization)** — preferido sobre DQN porque:
- Maneja acciones continuas (tamaño de posicion).
- Mas estable en convergencia.
- Implementacion robusta en `stable-baselines3`.

#### Tareas
1. Entrenar PPO en entorno simulado (train set).
2. Hyperparameter tuning: learning rate, n_steps, batch_size, γ.
3. Comparar contra:
   - Baseline 1: regla Z-score clasica (Fase 5).
   - Baseline 2: LSTM + regla deterministica (sin RL).
4. Curva de aprendizaje + analisis de politica aprendida.

---

### FASE 9 — Backtesting y Auditoria Final
**Entregable**: `09_backtest.py`, `reports/final_report.html`

#### Tareas
1. Backtest **out-of-sample** (2024-2026) con costos realistas.
2. Calcular **todos los KPIs** de la Seccion 1.3.
3. Analisis de robustez:
   - Bootstrap de los retornos (¿Sharpe es significativo?).
   - Sensibilidad a costos (1bp, 5bp, 10bp, 20bp).
   - Performance por regimen (HMM).
4. Comparar 3 estrategias: regla clasica vs LSTM+regla vs LSTM+RL.
5. Plots finales: equity curve, drawdown, rolling Sharpe, distribuciones de trade.

---

## 5. Stack Tecnico

| Capa | Librerias |
|---|---|
| **Datos** | yfinance, pandas, numpy |
| **Econometria** | statsmodels (ADF, Johansen, GARCH), arch, pykalman |
| **ML clasico** | scikit-learn, hmmlearn |
| **Deep Learning** | PyTorch (preferido) o TensorFlow/Keras |
| **RL** | gymnasium, stable-baselines3 |
| **Backtesting** | vectorbt o backtrader (opcional, o custom) |
| **Visualizacion** | matplotlib, plotly, seaborn |
| **Reporting** | jupyter, papermill, quarto (opcional) |

---

## 6. Estructura de Archivos Sugerida

```
project_final/
├── PLAN_PROYECTO.md              ← este documento
├── README.md                      ← guia rapida de ejecucion
├── requirements.txt
│
├── 01_data_ingestion.py          ✅
├── 02_cointegration.py           ✅
├── 03_dynamic_hedge.py           🔜
├── 04_features.py
├── 05_signals.py
├── 06_deep_model.py
├── 07_env_gym.py
├── 08_rl_agent.py
├── 09_backtest.py
│
├── data/
│   ├── gold_silver_panel.csv     ✅
│   ├── spread.csv                ✅
│   ├── kalman_spread.csv
│   ├── features.parquet
│   └── signals.csv
│
├── models/
│   ├── lstm_best.pt
│   └── ppo_agent.zip
│
├── envs/
│   └── PairTradingEnv.py
│
├── reports/
│   ├── normalized_prices.png     ✅
│   ├── spread_bands.png          ✅
│   └── final_report.html
│
└── notebooks/
    ├── eda.ipynb                 ← exploracion interactiva
    └── results_analysis.ipynb
```

---

## 7. Cronograma Sugerido (8 semanas)

| Semana | Fases | Entregable critico |
|---|---|---|
| 1 | Fase 1-2 | ✅ Datos validados + diagnostico cointegracion |
| 2 | Fase 3 | Kalman filter + cointegracion rolling |
| 3 | Fase 4 | Features completos (GARCH, HMM, tecnicos) |
| 4 | Fase 5 + 6 (inicio) | Benchmark clasico + setup LSTM |
| 5 | Fase 6 | LSTM entrenado y validado |
| 6 | Fase 7 | Entorno Gym funcional |
| 7 | Fase 8 | Agente PPO entrenado |
| 8 | Fase 9 | Backtest, KPIs, reporte final |

---

## 8. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| Cointegracion no se recupera ni con Kalman | Media | Alto | Pivotar a estrategia de regimen-switching pura sin asumir cointegracion |
| LSTM overfittea (Sharpe bueno en train, malo en test) | Alta | Alto | Walk-forward estricto, dropout, early stopping, regularizacion |
| RL no converge | Media | Medio | Empezar con DQN simple, luego PPO. Recompensa shaping conservadora |
| Costos de transaccion matan el Sharpe | Alta | Alto | Analisis de sensibilidad temprano (Fase 5), penalizar turnover en reward |
| Look-ahead bias accidental | Media | Critico | Auditoria de pipeline, test unitario que verifica que features en t solo usan datos ≤ t |
| Data leakage por scaling global | Alta | Critico | Fit scalers solo en train, aplicar a val/test |

---

## 9. Diferencias clave vs Proyecto Original

| Aspecto | Original | Refinado |
|---|---|---|
| **Hedge ratio** | β constante (OLS global) | β_t dinamico (Kalman) + validacion rolling |
| **Cointegracion** | Asumida | **Diagnosticada empiricamente** y adaptada al hallazgo |
| **Features** | EMA, GARCH, tecnicos | + HMM regimes, GSR Z-score, cross-asset opcional |
| **Validacion ML** | Walk-forward mencionado | Walk-forward + leak-checks + bootstrap |
| **Benchmark** | No explicito | Regla clasica Z-score como baseline obligatorio |
| **Costos** | Mencionados | Modelados explicitamente (5 bps) y con analisis de sensibilidad |
| **Reporting** | General | KPIs cuantificados con umbrales |

---

## 10. Proximos Pasos Inmediatos

1. **Crear `03_dynamic_hedge.py`** — implementar Kalman filter y cointegracion rolling.
2. Validar que el spread dinamico cumple estacionariedad en ventanas.
3. Si funciona → continuar a Fase 4 (features).
4. Si no → escalar a estrategia de regimen-switching (Plan B descrito en seccion 8).

---

## Apendice A — Glosario Tecnico Rapido

- **Cointegracion**: dos series no estacionarias cuya combinacion lineal SI es estacionaria.
- **Hedge ratio (β)**: cuanto de un activo abrir contra el otro para neutralizar riesgo direccional.
- **Spread**: el "residuo" ε_t = ln(P_oro) − β·ln(P_plata). El objeto a tradear.
- **Z-score**: cuantas desviaciones estandar esta el spread de su media. Señal de entrada/salida.
- **Walk-forward validation**: entrenar en [t-N, t], evaluar en [t, t+k], avanzar y repetir. Preserva causalidad temporal.
- **GARCH**: modelo de volatilidad condicional (volatilidad no constante).
- **HMM**: Hidden Markov Model — detecta regimenes latentes en la serie.
- **PPO**: Proximal Policy Optimization — algoritmo de RL state-of-the-art para control continuo.
- **Sharpe Ratio**: retorno medio / volatilidad. > 1 bueno, > 2 excelente.
- **Max Drawdown**: peor caida desde un maximo historico. Mide el "peor escenario".
- **Market-Neutral**: estrategia cuyo retorno no depende de la direccion general del mercado.
