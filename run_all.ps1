# Ejecuta el pipeline completo (Windows PowerShell)
# Uso:  .\run_all.ps1

$ErrorActionPreference = "Stop"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$Notebooks = @(
  "01_data_ingestion.ipynb",
  "02_cointegration.ipynb",
  "03_dynamic_hedge.ipynb",
  "04_features.ipynb",
  "05_signals.ipynb",
  "06_deep_model.ipynb",
  "07_env_gym.ipynb",
  "08_rl_agent.ipynb",
  "09_backtest.ipynb"
)

Write-Host "[INFO] Python:" (& $Python --version)
Write-Host "[INFO] Pipeline de $($Notebooks.Count) notebooks comenzando..."

foreach ($nb in $Notebooks) {
  Write-Host ""
  Write-Host "================================================================="
  Write-Host "  Ejecutando $nb"
  Write-Host "================================================================="
  & $Python -m jupyter nbconvert --to notebook --execute --inplace `
    --ExecutePreprocessor.timeout=1800 $nb
  if ($LASTEXITCODE -ne 0) { throw "Fallo $nb" }
}

Write-Host ""
Write-Host "[OK] Pipeline completo. Reporte final: data/final_metrics.csv"
