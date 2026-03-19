# FRAME-FM
**Framework for the Rapid development of Environmental Foundation Models**

FRAME-FM is an open-source software framework designed to enable the fast, scalable, and accessible development of Foundation Models (FMs) for large-scale environmental datasets, including petabyte-scale archives held by the UK’s NERC Environmental Data Service (EDS).

The project addresses a critical gap: while environmental data archives are vast and information-rich, they are difficult to process directly. FRAME-FM lowers the barrier to using these datasets by providing standardised workflows, infrastructure, and tools that allow users to train, evaluate, fine-tune, and publish foundation models efficiently.
<details>
  <summary><strong>📚 Table of Contents</strong></summary>

---

### 🔧 Core Documentation
- [Environment Setup](#environment-setup-from-pyprojecttoml)
- [Hydra Configuration System](./configs/configs_README.md)

### 🧠 Code Structure
- [Model Code](./src/FRAME_FM/model_code/models_README.md)
- [Data Loaders](./src/FRAME_FM/dataloaders/dataloaders_README.md)
- [Training Pipeline](./src/FRAME_FM/training/training_README.md)
- [Utilities](./src/FRAME_FM/utils/utils_README.md)

### 📊 Experiments & Logging
- [Experiment Configuration](./configs/experiment/experiment_README.md)
- [Logging (MLflow)](./configs/logging/logging_README.md)

### 📒 Notebooks
- [Jupyter Notebooks](./notebooks/notebooks_README.md)
- [Marimo Notebooks](./notebooks/marimo_notebooks_README.md)

### 📒 [Available Commands](#commands)
---

</details>



## Environment Setup from `pyproject.toml`
### 🚀 Setting Up the Environment (from pyproject.toml)

This project uses **uv** for dependency management.  
If you already have the repository (including `pyproject.toml` and `uv.lock`), use the steps below to recreate the full environment.

## 0. JASMIN
if your are running on the UK science compute service (jasmin.ac.uk)`
once logged in run
```bash
module load jaspy
```


## 1. Install uv

```bash
pip install uv
```

## 2. Create virutal environment and install dependencies with uv 

```bash
uv venv
uv sync
```

## 3. Add Additional Dependencies
Additional sources, such as `torchgeo` are installable sperately - so as to reduce wasting storage.

To install them, either run the relevant `uv add` command such as:
```bash
uv add torchgeo --optional data
```
OR you can install all sources using the `data` extra;
```bash
uv sync --extra data
```




---

<sub>⚠️ Some README files in this repository were generated using ChatGPT.  
All generated text has been manually reviewed to ensure accuracy and project-specific relevance.</sub>

---

## Commands

# `framefm train run`

Starts a model training run. Configuration is loaded from the `configs/` directory using Hydra. You can override any config value directly from the command line without editing any YAML files.

## Usage

```bash
framefm train run [OPTIONS] [OVERRIDES]...
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Print the fully resolved Hydra config to screen before training starts. |
| `--help` | | Show help and exit. |

## Config overrides

Any positional arguments after `train run` are passed to Hydra as config overrides.

| Syntax | Meaning | Example |
|--------|---------|---------|
| `key=value` | Override an existing value. Raises an error if the key does not exist. | `seed=99` |
| `+key=value` | Append a new key. Raises an error if the key already exists. | `+experiment=baseline` |
| `++key=value` | Override or append — safe either way. | `++new_key=99` |
| `~key=value` | Remove a key from the config. | `~logging=demo_mlflow` |

### Config groups

Hydra organises config files into groups (e.g. `model/`, `data/`). Passing a group name as an override swaps the entire file loaded for that group.

```
configs/
    config.yaml
    model/
        demo_autoencoder.yaml
        convAE.yaml
    data/
        demo_eurosat.yaml
        land_cover_map.yaml
    trainer/
        default.yaml
    logging/
        demo_mlflow.yaml
```

For example, `model=convAE` tells Hydra to load `configs/model/convAE.yaml` instead of the default.

## Examples

**Run with defaults**
```bash
framefm train run
```

**Override a config group**
```bash
framefm train run model=convAE
framefm train run data=land_cover_map
```

**Override multiple values**
```bash
framefm train run model=convAE data=land_cover_map seed=99
```

**Override a nested value**
```bash
framefm train run model=convAE model.lr=1e-4
```

**Append a new experiment config**
```bash
framefm train run +experiment=baseline
```

**Remove logging**
```bash
framefm train run ~logging=demo_mlflow
```

**Print the resolved config before training**
```bash
framefm train run --verbose model=convAE
```

## Default config

The root config file is `configs/config.yaml`. Its defaults list determines which group configs are loaded unless overridden on the command line:

```yaml
defaults:
  - data: demo_eurosat
  - model: demo_autoencoder
  - trainer: default
  - logging: demo_mlflow
  - _self_

seed: 42
```

## Further reading

- [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra config groups](https://hydra.cc/docs/tutorials/structured_config/config_groups/)





