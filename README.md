![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![GitHub Copilot](https://img.shields.io/badge/github_copilot-8957E5?style=for-the-badge&logo=github-copilot&logoColor=white)
 
 >Disclaimer: Some parts of this repo were generated with assistance from AI tools, such as ChatGPT and GitHub Copilot. All generated content has been reviewed for accuracy and relevance to the project.
----


# FRAME-FM
**Framework for the Rapid development of Environmental Foundation Models**

FRAME-FM is an open-source software framework designed to enable the fast, scalable, and accessible development of Foundation Models (FMs) for large-scale environmental datasets, including petabyte-scale archives held by the UK NERC Environmental Data Service (EDS).

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

---

</details>

## Environment Setup from `pyproject.toml`
### 🚀 Setting Up the Environment (from pyproject.toml)

This project uses **uv** for dependency management.  
If you already have the repository (including `pyproject.toml` and `uv.lock`), use the steps below to recreate the full environment.

## 1. Very quick-start (when running on JASMIN)

If your are running on the UK science compute service (`jasmin.ac.uk`), you can get started quickly.

Login to a JASMIN `sci` server (from the `login` node):

```bash
ssh <user>@sci-vm-01.jasmin.ac.uk
```

Run an interactive job on the Slurm cluster, selecting a GPU node:

```bash
srun --gres=gpu:1 --mem=192000 --partition=orchid --account=orchid --qos=orchid --time=03:00:00 --pty /bin/bash
```

Clone the FRAME-FM repository, enter the directory and checkout branch:

```bash
git clone https://<GITHUB_ACCESS_TOKEN>@github.com/NERC-EDS/FRAME-FM
cd FRAME-FM
git checkout adamwa/mvp
```

Source the common virtual environment:

```bash
source /gws/ssde/j25b/eds_ai/frame-fm/code/envs/frame-fm/bin/activate
```

Run the demo:

```bash
PYTHONPATH=src  python -m FRAME_FM.training.train --config-name=FRAME_FM_DEMO
```

Or look at the configuration for the demo:

```bash
PYTHONPATH=src python -m FRAME_FM.training.train --config-name=FRAME_FM_DEMO --cfg job
```

## 2. More in-depth installation (not re-using existing JASMIN virtualenv)

Assuming you are on any system (not JASMIN), you will need to create your own software environment. We recommend using the `uv` package to do this. You can 
install `uv` with `pip`:

```bash
pip install uv
```

Or using `curl` and installing in your `$HOME` directory:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Which will put `uv` in ~/.local/bin, so add that to your $PATH
# (and optionally to your ~/.bashrc file)
export PATH=$PATH:${HOME}/.local/bin
```

Now you want to create your own virtualenv and install the dependencies:

```bash
# Create a local `.venv` file in the current working directory
uv venv
# Install all dependencies into the `.venv` directory
uv sync
```

You may want to add additional dependencies, such as `torchgeo`. 
These are installable sperately - so as to reduce wasting storage.

To install them, either run the relevant `uv add` command such as:

```bash
uv add torchgeo --optional data
```

OR you can install all sources using the `data` extra;

```bash
uv sync --extra data
```

Now things are installed, activate the `venv`:

```bash
source .venv/bin/activate
```

Run the demo:

```bash
PYTHONPATH=src  python -m FRAME_FM.training.train --config-name=FRAME_FM_DEMO
```

Or look at the configuration for the demo:

```bash
PYTHONPATH=src python -m FRAME_FM.training.train --config-name=FRAME_FM_DEMO --cfg job
```
