# Informe final del proyecto

Proyecto: Arbitraje estadistico Oro/Plata con hedge ratio dinamico, LSTM y PPO  
Fecha de consolidacion: 2026-05-23  
Ventana vigente de datos: 2022-01-01 a 2026-05-23 (`end` exclusivo en yfinance)  
Periodo efectivo del panel: 2022-01-04 a 2026-05-22  

## 1. Pregunta objetivo

La pregunta central del proyecto es:

> Es posible construir una estrategia de arbitraje estadistico entre oro y plata que, usando un hedge ratio dinamico, modelos predictivos secuenciales y aprendizaje por refuerzo, produzca retornos out-of-sample ajustados por riesgo superiores a un benchmark clasico de z-score?

La pregunta tiene tres subpreguntas tecnicas:

1. La relacion oro/plata puede modelarse mejor con un beta dinamico que con un beta fijo?
2. Un modelo LSTM aprende informacion util sobre la direccion futura del spread?
3. Un agente PPO mejora la ejecucion frente a reglas deterministicas simples?

## 2. Hipotesis de trabajo

La hipotesis original sostiene que oro y plata comparten factores economicos comunes, pero que su relacion no es constante. Por eso el proyecto reemplaza un hedge ratio fijo por un hedge ratio dinamico estimado con Kalman Filter.

La hipotesis operacional queda asi:

> Si el spread oro/plata se calcula con beta dinamico y se opera con reglas de entrada/salida o politicas aprendidas, entonces deberia obtenerse una mejora de performance out-of-sample frente a modelos estaticos o politicas triviales.

## 3. Alcance vigente

La version actual del proyecto trabaja con una muestra corta:

- Inicio configurado: 2022-01-01.
- Fin configurado: 2026-05-23.
- Datos efectivos: 2022-01-04 a 2026-05-22.
- Instrumentos: futuros continuos de Yahoo Finance `GC=F` y `SI=F`.
- Frecuencia: diaria.
- Test final: 2025-01-02 a 2026-05-15.

Se eligio esta ventana para reducir tiempo de ejecucion y probar si los resultados mejoraban al concentrarse en el regimen reciente.

## 4. Plan metodologico

El proyecto se ejecuta como pipeline de nueve fases:

1. Ingesta y validacion de datos.
2. Diagnostico de cointegracion estatica.
3. Estimacion de hedge ratio dinamico con Kalman Filter.
4. Feature engineering.
5. Benchmark clasico con senales z-score.
6. Modelo predictivo LSTM.
7. Entorno de simulacion Gymnasium.
8. Entrenamiento PPO.
9. Backtest comparativo out-of-sample.

Ademas, se agregaron modulos reutilizables:

- `src/backtest.py`: contabilidad corregida del spread, costos y metricas.
- `src/splits.py`: splits temporales estrictos con purging.
- `envs/pair_trading_env.py`: entorno Gym actualizado con la misma contabilidad del backtest.

## 5. Metodologia detallada

### 5.1 Datos

La ingesta descarga precios ajustados diarios de:

- Oro: `GC=F`.
- Plata: `SI=F`.

Se aplica inner join por fecha para conservar solo dias comunes. Luego se calculan retornos logaritmicos diarios.

Resultados de validacion:

- Observaciones finales del panel: 1102.
- Volatilidad anualizada oro: 18.83%, dentro del rango esperado.
- Volatilidad anualizada plata: 41.11%, levemente por encima del rango esperado.
- Correlacion oro/plata: 0.7718, dentro del rango esperado.

La volatilidad elevada de plata es un primer hallazgo: la muestra 2022-2026 contiene un regimen mas agresivo que la muestra larga anterior.

### 5.2 Cointegracion estatica

Se estima una regresion OLS entre log-precios:

```text
log(Gold_t) = alpha + beta * log(Silver_t) + error_t
```

Luego se evalua estacionariedad del spread estatico con ADF y se contrasta con Johansen.

Resultados:

- ADF del spread estatico: p-valor 0.193981.
- Johansen Trace: rango de cointegracion 0.
- Johansen Max-Eigen: rango de cointegracion 0.

Conclusion:

No hay evidencia suficiente de cointegracion estatica en la muestra 2022-2026. Esto justifica usar una relacion dinamica.

### 5.3 Hedge ratio dinamico con Kalman

El modelo usa un estado latente:

```text
x_t = [alpha_t, beta_t]
```

y una ecuacion de observacion:

```text
log(Gold_t) = alpha_t + beta_t * log(Silver_t) + ruido_t
```

Parametros actuales:

- `delta = 1e-5`.
- `observation_variance = 0.001`.
- Warm-up analitico: 252 dias.

Resultados:

- Ventanas estacionarias del spread estatico: 2.4%.
- Ventanas estacionarias del spread Kalman: 100.0%.
- ADF global del spread Kalman post warm-up: p-valor 0.000000.

Interpretacion:

Kalman produce un spread mucho mas estacionario que el beta fijo. Sin embargo, esto debe leerse como evidencia a favor de un modelo estado-espacio adaptativo, no como prueba definitiva de cointegracion clasica. La sensibilidad de parametros Kalman sigue siendo una limitacion.

### 5.4 Feature engineering

Se construyen 15 features:

- `spread_z`
- `spread_ema_5`
- `spread_ema_20`
- `spread_vol_20`
- `garch_vol_gold`
- `garch_vol_silver`
- `rsi_spread_14`
- `bollinger_pct_b`
- `gsr_z_60`
- `hmm_state`
- `ret_5d_gold`
- `ret_5d_silver`
- `ret_20d_gold`
- `ret_20d_silver`
- `beta_kalman`

El dataset final de features tiene:

- 1038 filas.
- 20 columnas.
- Rango: 2022-03-30 a 2026-05-15.

El target del LSTM es binario:

```text
target = 1 si spread_kalman(t+5) - spread_kalman(t) > 0
target = 0 en caso contrario
```

Limitacion importante:

GARCH y HMM siguen ajustados globalmente. Esto reduce pureza metodologica porque sus parametros ven toda la muestra. Los splits y scalers ya se corrigieron, pero GARCH/HMM rolling o expanding quedan como mejora pendiente.

### 5.5 Benchmark z-score

La estrategia base opera el spread Kalman con z-score:

- Entra long spread si `spread_z < -2`.
- Entra short spread si `spread_z > 2`.
- Sale si `abs(spread_z) < 0.5`.
- Evita/stop si `abs(spread_z) >= 4`.

El backtest corregido:

- Normaliza PnL por exposicion bruta `1 + abs(beta)`.
- Cobra costos por turnover real de ambas piernas.
- Usa posicion rezagada para evitar look-ahead.

Resultado full sample de benchmark:

- Sharpe: 0.007.
- Max drawdown: -13.1%.
- Equity final: 0.990.
- Turnover acumulado: 77.17 veces capital.
- Costos acumulados: 3.86%.

El benchmark no es bueno en toda la muestra 2022-2026, pero mejora en el periodo test 2025+.

### 5.6 LSTM predictivo

El modelo usa:

- Lookback: 60 dias.
- Features: 15.
- LSTM hidden size: 32.
- Capas LSTM: 1.
- Head densa: 16 neuronas + ReLU + Dropout.
- Loss: CrossEntropy.
- Optimizer: Adam.
- Early stopping por validation loss.

Splits vigentes:

- Train purgado: 2022-03-30 a 2023-12-21, 436 filas antes de ventanas.
- Validation purgado: 2024-01-02 a 2024-12-23, 247 filas antes de ventanas.
- Test: 2025-01-02 a 2026-05-15, 345 filas antes de ventanas.

Ventanas efectivas para evaluacion:

- Train: 376.
- Validation: 187.
- Test: 285.

Metricas de clasificacion:

| Split | Accuracy | AUC | Brier |
|---|---:|---:|---:|
| Validation | 0.7219 | 0.7340 | 0.2042 |
| Test | 0.5825 | 0.6289 | 0.2518 |

Lectura:

El LSTM detecta cierta informacion direccional en test (AUC 0.6289), pero eso no se traduce en PnL positivo con la politica usada. Este es un hallazgo clave: predecir direccion no equivale automaticamente a una estrategia rentable.

### 5.7 Entorno Gymnasium

El entorno `PairTradingEnv` define:

- Acciones discretas:
  - 0: short spread.
  - 1: flat.
  - 2: long spread.
- Observacion:
  - 11 features de mercado/modelo.
  - posicion actual.
- Reward:
  - PnL neto del spread.
  - pesos normalizados por exposicion bruta.
  - costos por turnover real de las dos piernas.
  - penalizacion opcional de turnover.

Baselines del entorno sobre toda la muestra:

- Always Flat: equity 1.0000, Sharpe 0.000.
- Always Long: equity 0.5214, Sharpe -0.750.
- Always Short: equity 1.6557, Sharpe 0.748.
- Random: equity 0.5179, Sharpe -0.913.
- LSTM policy simple: equity 0.6818, Sharpe -0.547.

Lectura:

En la muestra reciente domina una direccion del spread favorable a short spread. Esto explica por que un baseline simple puede verse fuerte, pero tambien alerta sobre riesgo de dependencia a un solo regimen.

### 5.8 PPO

El agente PPO usa:

- Stable-Baselines3.
- Politica MLP `[64, 64]`.
- Timesteps: 100000.
- Acciones discretas.
- Reward corregido del entorno.

Resultados PPO:

- Validation Sharpe: -0.1865.
- Test Sharpe: 0.0188.
- Test final equity: 0.9979.
- En test, la politica estuvo mayormente flat: 290 dias flat, 52 short, 1 long.

Lectura:

PPO no aprende una politica activa superior. La politica resultante preserva capital casi plano, pero no mejora al benchmark z-score.

## 6. Backtest final out-of-sample

Periodo test:

- Inicio: 2025-01-02.
- Fin: 2026-05-15.
- Dias evaluados para benchmark/LSTM: 345.
- Dias evaluados para PPO: 343.

Resultados:

| Estrategia | Sharpe | Sortino | MaxDD | Calmar | WinRate activo | Equity final |
|---|---:|---:|---:|---:|---:|---:|
| Z-score Benchmark | 0.7617 | 0.4401 | -6.52% | 1.1500 | 43.33% | 1.1041 |
| LSTM Policy | -0.5063 | -0.5276 | -28.74% | -0.3477 | 47.14% | 0.8658 |
| PPO Agent | 0.0188 | 0.0098 | -6.74% | -0.0224 | 37.18% | 0.9979 |

Costos y turnover:

| Estrategia | Turnover | Costos totales | Exposicion bruta promedio |
|---|---:|---:|---:|
| Z-score Benchmark | 27.05x | 1.35% | 2.47x |
| LSTM Policy | 55.17x | 2.76% | 2.47x |
| PPO Agent | No reportado en tabla final | No reportado en tabla final | No reportado en tabla final |

Sensibilidad a costos:

| Costo bps | Benchmark Sharpe | LSTM Sharpe | PPO Sharpe |
|---:|---:|---:|---:|
| 0 | 0.8550 | -0.3924 | 0.2606 |
| 1 | 0.8364 | -0.4152 | 0.2121 |
| 2 | 0.8178 | -0.4379 | 0.1636 |
| 5 | 0.7617 | -0.5063 | 0.0188 |
| 10 | 0.6674 | -0.6200 | -0.2207 |
| 20 | 0.4757 | -0.8465 | -0.6888 |
| 50 | -0.1148 | -1.5101 | -1.9595 |

## 7. Hallazgos principales

### Hallazgo 1: no hay cointegracion estatica robusta

El beta fijo no produce un spread suficientemente estacionario. Tanto ADF como Johansen rechazan la idea de una relacion estatica fuerte en la muestra 2022-2026.

### Hallazgo 2: Kalman mejora fuertemente la estacionariedad del spread

El spread Kalman pasa a ser estacionario en 100% de ventanas ADF evaluadas. Esto valida la decision metodologica de usar beta dinamico, aunque falta sensibilidad formal de hiperparametros.

### Hallazgo 3: el benchmark z-score es la mejor estrategia out-of-sample

En test 2025+, el benchmark obtiene:

- Sharpe 0.7617.
- Calmar 1.15.
- MaxDD -6.52%.
- Equity 1.1041.

No cumple Sharpe >= 1.5, pero es claramente la mejor de las tres estrategias comparadas.

### Hallazgo 4: LSTM clasifica razonablemente, pero opera mal

El LSTM obtiene AUC test 0.6289. Aun asi, la politica basada en umbral `0.55 / 0.45` pierde dinero. La razon probable es una combinacion de:

- exceso de exposicion,
- alto turnover,
- mala calibracion de umbrales,
- desconexion entre target direccional y utilidad economica.

### Hallazgo 5: PPO no agrega valor en esta configuracion

PPO queda casi plano. No supera al benchmark ni aprende una politica suficientemente activa. El agente parece preferir evitar riesgo mas que explotar senales.

### Hallazgo 6: la ventana corta cambia la lectura del proyecto

La muestra reciente favorece estrategias short spread o mean-reversion especifica del periodo. Esto mejora el benchmark en test, pero reduce la confianza en generalizacion fuera de este regimen.

## 8. Evaluacion contra KPIs

| KPI | Objetivo | Mejor resultado actual | Estado |
|---|---:|---:|---|
| Sharpe anualizado | >= 1.5 | 0.7617 | No cumple |
| Sortino | >= 2.0 | 0.4401 | No cumple |
| Max drawdown | <= 15% | -6.52% benchmark | Cumple para benchmark |
| Calmar | >= 1.0 | 1.1500 benchmark | Cumple para benchmark |
| Beta vs S&P500 | abs(beta) <= 0.2 | No calculado | Pendiente |
| Win rate activo | >= 55% | 47.14% LSTM | No cumple |
| Spread estacionario rolling | p < 0.05 en mayoria | 100% Kalman | Cumple |

Conclusion sobre KPIs:

El proyecto cumple parcialmente en riesgo y estacionariedad, pero no cumple la meta principal de Sharpe ni la validacion completa de market-neutrality.

## 9. Calidad tecnica actual

Puntos fuertes:

- Pipeline completo de 9 notebooks ejecutado.
- Artefactos regenerables en `data/` y `models/`.
- Backtest corregido por exposicion bruta.
- Costos por turnover real en benchmark/LSTM.
- Splits temporales purgados.
- Tests vigentes: 46 casos.
- Entorno Gym compatible con Gymnasium.

Puntos debiles:

- GARCH y HMM siguen ajustados globalmente.
- No hay walk-forward completo.
- No hay bootstrap ni intervalos de confianza del Sharpe.
- No se calcula beta vs S&P500.
- El PPO no reporta turnover/costos en la tabla final.
- Los notebooks siguen conteniendo gran parte de la logica.
- `params.yaml` documenta configuracion, pero los notebooks aun tienen valores inline.

## 10. Estado de artefactos

Artefactos principales vigentes:

- `data/gold_silver_panel.csv`: panel base 2022-2026.
- `data/spread_static.csv`: spread con beta fijo.
- `data/kalman_spread.csv`: alpha, beta y spread dinamico.
- `data/features.parquet`: features finales.
- `data/features_with_lstm.parquet`: features + probabilidades LSTM.
- `data/signals_benchmark.csv`: senales y PnL benchmark.
- `data/final_metrics.csv`: tabla final de metricas.
- `data/final_report.json`: reporte final estructurado.
- `models/lstm_best.pt`: pesos LSTM.
- `models/ppo_agent.zip`: agente PPO.

## 11. Veredicto

El proyecto es un prototipo academico completo y tecnicamente funcional. La metodologia actual es mucho mas defendible que la version inicial porque corrige contabilidad del spread, costos y splits temporales.

La respuesta a la pregunta objetivo es:

> No se encontro evidencia suficiente de que LSTM o PPO mejoren al benchmark z-score. En la ventana 2022-2026, el benchmark clasico es la mejor estrategia out-of-sample, aunque tampoco alcanza el Sharpe objetivo de 1.5.

El hallazgo mas importante no es que el ML gane, sino que no gana bajo una evaluacion mas realista. Eso es una conclusion valida para una entrega academica: el proyecto demuestra el proceso completo y reporta honestamente que los modelos complejos no superan una regla simple en esta muestra.

## 12. Recomendaciones

Prioridad alta:

1. Implementar GARCH/HMM rolling o expanding para eliminar leakage residual.
2. Implementar walk-forward real con reentrenamiento por ventana.
3. Optimizar umbrales de LSTM en validation con objetivo economico, no solo AUC.
4. Reportar turnover y costos PPO en la tabla final.
5. Calcular beta vs S&P500 para validar market-neutrality.

Prioridad media:

1. Agregar bootstrap del Sharpe.
2. Agregar Probabilistic Sharpe Ratio o Deflated Sharpe Ratio.
3. Probar position sizing proporcional a confianza.
4. Redisenar reward PPO con penalizacion por drawdown/volatilidad.
5. Extraer mas logica de notebooks a `src/`.

Prioridad baja:

1. Evaluar otros pares de metales para diversificacion.
2. Probar modelos GRU o Transformer pequeno.
3. Incorporar slippage variable por volatilidad.

## 13. Conclusion final

El sistema actual debe presentarse como:

- Un pipeline completo de investigacion cuantitativa.
- Un caso de estudio de arbitraje estadistico oro/plata.
- Una comparacion honesta entre benchmark clasico, LSTM y PPO.
- Una evidencia de que mas complejidad no garantiza mejor PnL.

No debe presentarse como:

- Una estrategia lista para trading real.
- Una estrategia que cumple Sharpe objetivo.
- Una prueba definitiva de cointegracion clasica.
- Una validacion completa de market-neutrality.

El resultado final es defendible si se comunica con esta lectura: el benchmark z-score sobre spread Kalman es el unico enfoque que muestra performance positiva en test 2025+, mientras que LSTM y PPO requieren redisenio antes de poder considerarse mejoras reales.
