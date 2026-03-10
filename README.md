# FRAME-FM

## Overview

### Purpose

Welcome to the **Framework for the Rapid Developement of Environmental Foundation Models** (FRAME-FM).

FRAME-FM is an open-source software framework designed to enable the fast, scalable, and accessible development of Foundation Models (FMs) for large-scale environmental datasets. This includes petabyte-scale archives held by the UK’s NERC Environmental Data Service (EDS).

While environmental data archives are vast and information-rich, they are difficult to process directly. This project addresses this critical gap by lowering the barrier to use these datasets by providing standardised workflows, infrastructure, and tools. This allow users to train, evaluate, fine-tune, and publish foundation models efficiently.

### Current Status

This repository was created in October 2025. As such, it is quickly evolving as more functionality, documentation and examples are provided.

- What is it's current state?
- What is and isn't complete?

### Repository Breakdown

- Structure of the Repo
- Current state (what is complete and what isn't complete)

### Table of Contents

- [Who is FRAME-FM Built for?](#who-is-frame-fm-built-for)
- [Running FRAME-FM Locally](#running-frame-fm-locally)
- [Running FRAME-FM on JASMIN](#running-frame-fm-on-jasmin)
- [Other Examples](#other-examples)
- [Technical Documentation](#technical-documentation)

---

## Who is FRAME-FM Built for?

- Scientists who...

---

## Running FRAME-FM Locally

### Pre-Requisites

- Designed for Linux
- Python >=3.11,<3.13
- uv

### Hardware Requirements

### Steps

### Expected Output

---

## Running FRAME-FM on JASMIN

### Pre-Requisites

- JASMIN access

---

## Other Examples

---

## Technical Documentation

---










































---

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


