# Guion de exposicion - Proyecto ML Oro/Plata

Proyecto: Arbitraje estadistico Oro/Plata con hedge ratio dinamico, LSTM y PPO  
Duracion sugerida: 18 a 24 minutos  
Equipo: 3 personas  
Datos vigentes: 2022-01-04 a 2026-05-22  
Test final: 2025-01-02 a 2026-05-15  

## Distribucion general

- Persona 1: contexto, pregunta objetivo, datos, cointegracion y Kalman. Diapositivas 1 a 8.
- Persona 2: machine learning, features, LSTM, entorno Gym y PPO. Diapositivas 9 a 16.
- Persona 3: backtest, resultados, KPIs, limitaciones, reproducibilidad y cierre. Diapositivas 17 a 24.

## Diapositiva 1 - Titulo y tesis del proyecto

Responsable: Persona 1

Texto sugerido:

"Buenos dias, profesor. Nuestro proyecto se llama Arbitraje estadistico Oro/Plata con cointegracion dinamica, LSTM y aprendizaje por refuerzo. La idea central fue construir un pipeline completo de investigacion cuantitativa: primero analizamos si oro y plata mantienen una relacion estadistica operable; luego modelamos esa relacion con un hedge ratio dinamico; despues agregamos machine learning para predecir el spread; y finalmente comparamos tres estrategias en un backtest out-of-sample. La conclusion principal es honesta: el benchmark clasico con z-score fue el mejor en test, mientras que LSTM y PPO no lograron superarlo como estrategias de trading."

## Diapositiva 2 - Pregunta objetivo

Responsable: Persona 1

Texto sugerido:

"La pregunta objetivo fue: es posible construir una estrategia de arbitraje estadistico entre oro y plata que, usando beta dinamico, modelos secuenciales y aprendizaje por refuerzo, mejore los retornos ajustados por riesgo frente a una regla clasica de z-score? Esta pregunta se divide en tres partes. Primero, si el beta dinamico mejora al beta fijo. Segundo, si un LSTM aprende informacion predictiva sobre la direccion futura del spread. Tercero, si PPO puede convertir esas senales en mejores decisiones de ejecucion."

## Diapositiva 3 - Hipotesis financiera

Responsable: Persona 1

Texto sugerido:

"La hipotesis financiera parte de que oro y plata comparten factores macroeconomicos: inflacion, tasas reales, fortaleza del dolar y demanda de metales. Sin embargo, no asumimos que la relacion sea fija. La plata tiene una componente industrial mas fuerte, por eso el ratio oro/plata puede cambiar de regimen. Por esa razon, el proyecto no se queda con una regresion estatica; usa un Kalman Filter para estimar alpha y beta variables en el tiempo."

## Diapositiva 4 - Arquitectura del pipeline

Responsable: Persona 1

Texto sugerido:

"El pipeline tiene nueve fases. Descargamos datos, probamos cointegracion estatica, estimamos el spread dinamico con Kalman, construimos features, generamos un benchmark con z-score, entrenamos un LSTM, definimos un entorno Gym, entrenamos PPO y finalmente hacemos el backtest comparativo. Tambien agregamos modulos reutilizables: backtest.py para contabilidad corregida, splits.py para cortes temporales purgados, y pair_trading_env.py para el entorno de reinforcement learning."

## Diapositiva 5 - Datos e ingesta

Responsable: Persona 1

Texto sugerido:

"La ventana vigente fue recortada a 2022-2026 para reducir tiempo de ejecucion y enfocarnos en el regimen reciente. Usamos futuros continuos de Yahoo Finance: GC=F para oro y SI=F para plata. El panel efectivo tiene 1102 observaciones, desde el 4 de enero de 2022 hasta el 22 de mayo de 2026. La volatilidad anualizada del oro fue 18.83%, dentro del rango esperado; la plata tuvo 41.11%, un poco por encima del rango esperado; y la correlacion de retornos fue 0.7718, que si esta dentro de lo esperado para estos metales."

## Diapositiva 6 - Diagnostico de cointegracion estatica

Responsable: Persona 1

Texto sugerido:

"Primero probamos la metodologia clasica: una regresion OLS entre log-precios de oro y plata, y luego un ADF sobre el spread. El p-valor del ADF del spread estatico fue 0.193981, por lo tanto no rechazamos la hipotesis nula de raiz unitaria. Tambien aplicamos Johansen, y tanto Trace como Max-Eigen dieron rango de cointegracion cero. Esto significa que, para esta muestra, un beta fijo no produce una relacion estacionaria suficientemente fuerte."

## Diapositiva 7 - Kalman Filter y beta dinamico

Responsable: Persona 1

Texto sugerido:

"La solucion fue estimar un beta dinamico con Kalman Filter. El estado latente tiene alpha_t y beta_t, y la observacion es log del oro como funcion de log de la plata. Esto permite que la relacion se adapte cada dia. El resultado fue muy fuerte: el spread estatico fue estacionario en solo 2.4% de las ventanas, mientras que el spread Kalman fue estacionario en 100% de las ventanas evaluadas. Esto valida la decision de modelar la relacion como dinamica, aunque no debe interpretarse como una prueba definitiva de cointegracion clasica."

## Diapositiva 8 - Transicion hacia Machine Learning

Responsable: Persona 1

Texto sugerido:

"Hasta aqui tenemos el objeto central del proyecto: un spread dinamico mas estacionario. La pregunta siguiente es si podemos predecir sus movimientos y operar mejor. Aqui entra la parte de machine learning. Le paso la palabra a mi companero, que explicara las features, el LSTM, el entorno Gym y el agente PPO."

## Diapositiva 9 - Feature engineering para ML

Responsable: Persona 2

Texto sugerido:

"La parte de machine learning empieza con el diseno de features. Construimos 15 variables que resumen el estado del spread y del mercado. Incluyen z-score del spread, medias exponenciales, volatilidad realizada, volatilidad GARCH de oro y plata, RSI, Bollinger percent B, z-score del gold-silver ratio, estado HMM, momentum de 5 y 20 dias, y beta Kalman. El dataset final tiene 1038 filas y 20 columnas, desde el 30 de marzo de 2022 hasta el 15 de mayo de 2026."

## Diapositiva 10 - Target, splits y purging

Responsable: Persona 2

Texto sugerido:

"El target del LSTM es direccional: vale 1 si el spread Kalman sube dentro de 5 dias, y 0 si no sube. Para evitar fuga temporal, usamos splits estrictos con purging. El entrenamiento cubre 2022-2023, validacion 2024, y test 2025-2026. Ademas, eliminamos las ultimas filas de train y validacion cuyo target cruzaba a la siguiente particion. Esto es importante porque en series financieras un pequeno leakage puede inflar mucho las metricas."

## Diapositiva 11 - Arquitectura LSTM

Responsable: Persona 2

Texto sugerido:

"El modelo predictivo fue un LSTM. Recibe una ventana de 60 dias por 15 features. La arquitectura tiene una capa LSTM con hidden size 32, seguida por una cabeza densa de 16 neuronas, ReLU, dropout y una capa final de dos clases. Usamos CrossEntropyLoss, Adam, regularizacion L2 y early stopping sobre validation loss. La idea es que el LSTM capture patrones temporales que una regla z-score no ve directamente."

## Diapositiva 12 - Resultados predictivos del LSTM

Responsable: Persona 2

Texto sugerido:

"En clasificacion, el LSTM si aprende algo. En validacion obtuvo accuracy 0.7219 y AUC 0.7340. En test bajo a accuracy 0.5825 y AUC 0.6289. Esto sigue estando por encima de azar, pero ya muestra degradacion out-of-sample. El punto clave es que una metrica predictiva positiva no garantiza rentabilidad. Cuando convertimos la probabilidad en posiciones, la estrategia LSTM pierde dinero."

## Diapositiva 13 - Politica LSTM como estrategia

Responsable: Persona 2

Texto sugerido:

"La politica LSTM fue sencilla: si la probabilidad de subida del spread era mayor a 0.55, tomaba long spread; si era menor a 0.45, tomaba short spread; si no, quedaba flat. En test estuvo activa cerca de 59.7% del tiempo, con turnover 55.17 veces capital. Ese exceso de actividad, junto con mala calibracion economica del umbral, hizo que terminara con equity 0.8658 y Sharpe -0.5063. Este es uno de los aprendizajes mas importantes del proyecto."

## Diapositiva 14 - Entorno Gymnasium

Responsable: Persona 2

Texto sugerido:

"Para reinforcement learning construimos un entorno custom con Gymnasium. La observacion incluye spread_z, beta Kalman, volatilidades, HMM, RSI, Bollinger, GSR, probabilidad LSTM y posicion actual. Las acciones son discretas: short spread, flat o long spread. Algo importante es que corregimos la contabilidad: el PnL se normaliza por exposicion bruta, 1 mas valor absoluto de beta, y los costos se cobran por turnover real de las dos piernas."

## Diapositiva 15 - Agente PPO

Responsable: Persona 2

Texto sugerido:

"El agente PPO es el componente de reinforcement learning. PPO significa Proximal Policy Optimization, y lo usamos desde Stable-Baselines3 con una politica MLP de dos capas de 64 neuronas. El agente interactua con el entorno: observa el estado, elige long, short o flat, recibe reward por PnL neto de costos, y actualiza su politica. Lo entrenamos por 100 mil timesteps. En test, la politica estuvo principalmente flat: 290 dias flat, 52 short y solo 1 long."

## Diapositiva 16 - Resultado ML y lectura critica

Responsable: Persona 2

Texto sugerido:

"La conclusion de machine learning es matizada. El LSTM aprende senal predictiva, porque el AUC test es 0.6289, pero esa senal no se transforma bien en PnL. PPO, por su parte, no aprende una politica que supere al benchmark. Esto no es un fracaso del proyecto: es una conclusion empirica. En finanzas, mas complejidad no necesariamente mejora una regla simple. Ahora la tercera parte mostrara el backtest final y la comparacion de estrategias."

## Diapositiva 17 - Backtest corregido

Responsable: Persona 3

Texto sugerido:

"El backtest final compara tres estrategias: benchmark z-score, LSTM policy y PPO agent. La parte critica fue corregir la contabilidad. Antes, el PnL podia sobreestimar resultados porque no normalizaba por exposicion bruta. Ahora cada posicion se convierte en pesos de oro y plata, se divide por 1 mas abs(beta), y los costos se calculan por turnover real. Tambien se usa posicion rezagada para evitar look-ahead."

## Diapositiva 18 - Resultados finales out-of-sample

Responsable: Persona 3

Texto sugerido:

"El test va del 2 de enero de 2025 al 15 de mayo de 2026. El benchmark z-score fue el mejor: Sharpe 0.7617, Sortino 0.4401, drawdown maximo -6.52%, Calmar 1.15 y equity final 1.1041. La politica LSTM tuvo Sharpe -0.5063 y equity 0.8658. PPO quedo casi plano, con Sharpe 0.0188 y equity 0.9979. Por tanto, en esta muestra, el metodo clasico gana."

## Diapositiva 19 - Sensibilidad a costos

Responsable: Persona 3

Texto sugerido:

"Tambien hicimos sensibilidad a costos. El benchmark es relativamente robusto: con 0 bps tiene Sharpe 0.855, y con 5 bps queda en 0.762. Incluso con 20 bps sigue positivo, con Sharpe 0.476. LSTM es negativo en todos los costos evaluados, y PPO cae rapidamente cuando aumentan los costos. Esto confirma que el turnover de las estrategias ML fue un problema economico."

## Diapositiva 20 - Evaluacion contra KPIs

Responsable: Persona 3

Texto sugerido:

"Ahora evaluamos contra los objetivos. El proyecto no cumple el Sharpe objetivo de 1.5; el mejor fue 0.7617. Tampoco cumple Sortino 2.0 ni win rate activo de 55%. Si cumple Max Drawdown menor a 15% para el benchmark, y cumple Calmar mayor a 1 para el benchmark. La estacionariedad del spread Kalman si cumple. Queda pendiente beta contra S&P500 para validar market-neutrality completa."

## Diapositiva 21 - Hallazgos

Responsable: Persona 3

Texto sugerido:

"Los hallazgos principales son seis. Primero, no hay cointegracion estatica robusta. Segundo, Kalman mejora fuertemente la estacionariedad. Tercero, el benchmark z-score es la mejor estrategia out-of-sample. Cuarto, LSTM clasifica razonablemente pero opera mal. Quinto, PPO no agrega valor en esta configuracion. Sexto, la ventana corta favorece el benchmark, pero eso tambien significa que debemos ser cuidadosos con la generalizacion."

## Diapositiva 22 - Limitaciones

Responsable: Persona 3

Texto sugerido:

"Las limitaciones son importantes. GARCH y HMM siguen ajustados globalmente, por lo que queda un leakage metodologico residual. No implementamos walk-forward completo con reentrenamiento por ventana. No hay bootstrap ni intervalo de confianza del Sharpe. No se calcula beta vs S&P500. PPO no reporta turnover y costos en la tabla final. Y aunque ya movimos backtest y splits a src, los notebooks todavia concentran bastante logica."

## Diapositiva 23 - Reproducibilidad y artefactos

Responsable: Persona 3

Texto sugerido:

"El proyecto es reproducible con los notebooks en orden. Los artefactos principales son el panel de datos, el spread estatico, el spread Kalman, features.parquet, features_with_lstm.parquet, signals_benchmark.csv, final_metrics.csv y final_report.json. Los modelos entrenados quedan en models: lstm_best.pt y ppo_agent.zip. La suite actual tiene 46 tests que cubren imports, entorno, determinismo, estructura, splits purgados y contabilidad del backtest."

## Diapositiva 24 - Cierre

Responsable: Persona 3

Texto sugerido:

"La respuesta final a la pregunta objetivo es: no encontramos evidencia suficiente de que LSTM o PPO superen al benchmark z-score. El proyecto si demuestra un pipeline completo de investigacion cuantitativa y una mejora clara al usar Kalman para construir un spread mas estacionario. Pero como estrategia de trading, el mejor resultado sigue siendo la regla clasica. La conclusion academica es valiosa: mas machine learning no garantiza mas PnL, y una evaluacion honesta debe reportar cuando el modelo complejo no gana."

## Cierre conjunto para preguntas

Responsable: Persona 1, Persona 2 y Persona 3

Texto sugerido:

"Con esto cerramos la presentacion. Estamos listos para preguntas sobre la parte econometrica, la parte de machine learning o la evaluacion de backtesting."
