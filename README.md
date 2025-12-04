# FRAME-FM

## Environment Setup from `pyproject.toml`
### 🚀 Setting Up the Environment (from pyproject.toml)

This project uses **uv** for dependency management.  
If you already have the repository (including `pyproject.toml` and `uv.lock`), use the steps below to recreate the full environment.

## 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies with uv

```bash
pip install uv
uv sync
```