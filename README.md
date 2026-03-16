# FRAME-FM

## Overview

### Purpose

Welcome to the **Framework for the Rapid Developement of Environmental Foundation Models**
(FRAME-FM)!

FRAME-FM is an open-source software framework designed to enable the fast, scalable, and
accessible development of Foundation Models (FMs) for large-scale environmental datasets.
This includes petabyte-scale archives held by the UK’s NERC Environmental Data Service (EDS).

While environmental data archives are vast and information-rich, they are difficult to process
directly. This project addresses this critical gap by lowering the barrier to use these datasets.
FRAME-FM provides standardised workflows, infrastructure, and tools to allow users to train,
evaluate, fine-tune and publish foundation models efficiently.

### Table of Contents

- [Who is FRAME-FM Built for?](#who-is-frame-fm-built-for)
- [Running FRAME-FM Locally](#running-frame-fm-locally)
- [Running FRAME-FM on JASMIN](#running-frame-fm-on-jasmin)
- [Other Examples](#other-examples)
- [Technical Documentation](#technical-documentation)

---

## Who is FRAME-FM Built for?

FRAME-FM is intended for scientists who are both experienced and new to Machine Learning.

For those experienced in Machine Learning, FRAME-FM speeds up the data wrangling processes
by wrapping data loaders and setting up a thorough Extract ➡️ Transform ➡️ Load (ETL) pipeline.
This ETL pipeline also has data caching to lower resource usage and speed up the models.

For scientists that are new to Machine Learning, FRAME-FM provides a great starting point to
gain experience in Machine Learning. FRAME-FM abstracts away from needing to understand about
schedulers and settings. Instead, you can gain confidence with FRAME-FM and develop all the
knowledge to continue using PyTorch and Hydra in future projects either within our outside of
FRAME-FM.

---

## Repository Breakdown

The FRAME-FM repository contains many directories. The core of which are explained
below alongside details on the dependencies of FRAME-FM.

#### Essential Dependencies

| **Dependency** | **Purpose** |
|---|---|
| PyTorch Lightning | A high-level wrapper around PyTorch. It is used to build and train the foundation models. |
| Hydra | Manages the configuration for FRAME-FM by allowing config files to be written. |
| Mlflow | Allows recording and tracking runs either via a web GUI or through output logs. |

#### Source Code

All of the source code for the FRAME-FM project can be found in "src/FRAME-FM".

More specifically, this directory contains:

| **Subfolder** | **Description** |
|---|---|
| utils | This directory contains wrappers around important PyTorch modules. Wrappers like ` LightningDataModule ` and ` LightningModule ` contain customisations essential for FRAME-FM such as standardisations, constructions and consistent logging. |
| dataloaders | The ` LightningDataModule ` implementations for loading and pre-processing external datasets. This allows FRAME-FM to keep the model code separate from the data-handling, transform and batching logic. |
| datasets | Contains dataset wrappers to make applying transormations to input data easier. |
| models | The implementations of ` LightningModule ` from PyTorch. Each of the files within defines a modular and configurable model that should be instantiated through Hydra. |
| training | This holds the Hydra-driver ` train.py ` script used to launch model training runs. This directory also holds training entry points for Mlflow logging helpers, callbacks and trainer utilities. |
| transforms | This directory will hold all of our transformation classes and relevant utilities to glue them into FRAME-FM. |

---

## Running FRAME-FM

### Hardware Requirements

Although FRAME-FM can be run within Jupyter / Marimo notebooks or within Windows, it is designed
to be run on Linux. Although any computer can run FRAME-FM, it is intended to be used on supercomputer
platforms like EDS' JASMIN or the University of Bristol's Isembard. FRAME-FM is also intended for
x86 machines, but has successfully run in ARM-based machines.

Machine Learning is generally GPU-intensive. FRAME-FM's Extract ➡️ Transform ➡️ Load (ETL) pipeline
allows for data caching to minimise CPU, IO and memory usage. This minimises unnecessary pauses in
the GPU's execution.

### Pre-Requisites

The below guidance assumes that you already have all access required to connect and use
[JASMIN](https://accounts.jasmin.ac.uk/services/login_services/jasmin-login/).

If you are not using JASMIN, you will need to make sure that you have Python 3.11-3.13 and Pip
installed.

This documentation is intended to be run within Linux. However, with some small tweaks the below
commands can be run in Windows using WSL or Git Bash.

### Steps

Below are the steps to run *train.py* using the default configuration:

1. **SSH onto JASMIN**

    Once you have access to JASMIN, you should be able to set up a jump host (` -J `) to SSH onto
    one of the Sci servers through JASMIN's login servers. To do this, run:
    
    ```bash
    ssh -A <username>@sci-vm-04.jasmin.ac.uk -J <username>@login.jasmin.ac.uk
    ```

2. **Load the required module**

    Once connected, load the required modules with:

    ```bash
    module load jaspy
    ```

3. **Set up your Python path**

    Set up your python path by running the following commands:

    ```bash
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    ```

    When running ` python --version `, it should show Python 3.12.

4. **Set up a virtual environment**

    Next, you need to use UV to install a virtual environment (venv). This will install dependencies
    specifically within that environment.

    ```bash
    pip install uv
    ```

    ```bash
    uv venv
    ```

    ```bash
    source .venv/bin/activate
    ```

5. **Install all dependencies**

    There are a list of dependencies that need to be installed. These are visible in *pyproject.toml*.

    ```bash
    uv sync
    ```

    ```bash
    uv add torchgeo --optional data
    ```

6. **Running the training model**

    At this point, all of your dependencies are installed and your environment is set up. Run the
    default config with:

    ```bash
    python src/FRAME_FM/training/train.py
    ```

    If you would like to use Mlflow to record the training output, follow the
    [Logging README](./src/FRAME_FM/docs/logging_README.md). Note that this runs 4 epochs, so could
    take 2+ hours to complete. For a quick run, you can decrease the epochs in "configs/trainer/*default.yaml*".

---

## Technical Documentation

FRAME-FM uses Sphinx to generate and store technical documentation. This is a mixture of hand-written
and autogenerated documentation. For example, thorough details on the configs and transformers.

To view this documentation, visit https://nerc-eds.github.io/FRAME-FM/.

---
