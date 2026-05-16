# Contributing

## Setup

```bash
git clone <repo-url>
cd project_final
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements-dev.txt
```

## Workflow

1. Crear branch desde `main`: `git checkout -b feat/mi-cambio`
2. Hacer cambios + tests
3. Correr tests: `pytest tests/ -v`
4. Si modificas notebooks, re-ejecutarlos completos: `bash run_all.sh` (Linux/Mac) o `.\run_all.ps1` (Windows)
5. Commit (mensajes en imperativo: "add X", "fix Y")
6. PR a `main` con descripcion clara

## Reglas

- **Seeds**: cualquier nuevo modelo debe usar `random_state=42` o equivalente.
- **Anti-leakage**: features causales (rolling/EMA/shift). Standardizers `fit` solo en train.
- **Costos**: backtest siempre con costos realistas (no comparar gross PnL).
- **Tests**: cambios en `envs/` requieren test en `tests/test_env.py`.
- **Notebooks**: re-ejecutar y commitear con outputs limpios o vaciados — acordar con el equipo.

## Pre-commit (opcional)

```bash
pip install pre-commit
pre-commit install
```
