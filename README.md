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

Machine Learning is generally GPU-intensive and GPUs are typically more costly than CPUs. The 
FRAME-FM Extract ➡️ Transform ➡️ Load (ETL) pipeline is built to enable pre-processing and caching 
(which are IO and/or CPU-intensive) to happen before training/inference happens on GPUs. This approach 
is intended to optimise the use of GPUs when actual machines learning is taking place.

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

6. **Initialise configs**

    By default, FRAME-FM expects a `configs` directory to be present in your current working directory.
    Run the following command to initialise a configs directory if one does not already exist.
    (Note that you can override this location by setting the `CONFIG_DIR` environment variable. If set, the init command will use this path instead):

    ```bash
    uv run framefm config init
    ```

7. ** Configure Training **

   Edit the file `configs/config.yaml` and choose your platform. Platform configs are stored in `configs/platforms/<platform name.yaml>`.
   By default the "jasmin-small" config is used which allocates a single GPU and 2 CPU cores on JASMIN's Orchid cluster.

8. ** Run a Training Job **

    ```bash
    framefm train run
    ```

9. ** Monitoring Training with MLFlow **

    If you would like to use Mlflow to record the training output, follow the
    [Logging README](./src/FRAME_FM/docs/logging_README.md).


---
## Using the CLI
The torchx and hydra configs can be viewed and edited during runtime via the CLI. The training can also be run via CLI passing options and overrides.

## Config commands
| Command | Arguments | Flags | Description |
| :--- | :--- | :--- | :--- |
| `framefm config list` | None | `--torchx` | Recursively lists all YAML files in `configs/`. If `--torchx` is used, only verifies the `.torchxconfig` location. |
| `framefm config display` | `CONFIG_FILE` | `--torchx` | Prints the contents of a specific file. Use `--torchx` to skip the file argument and view the TorchX config. |
| `framefm config view-defaults` | None | None | Displays the top-level Hydra default configurations from `configs/config.yaml`. |
| `framefm config edit` | `FILE` `KV_PAIRS` | None | Updates YAML values. Format: `key:value` or `key:val1,key2:val2`. **Note:** Overwrites formatting and removes comments. |
| `framefm config edit-torchx` | `KV_PAIRS` | None | Updates `.torchxconfig`. Format: `<table>-<key>:<value>` (e.g., `defaults-cpu:4` to update the CPU value in the defaults section). |


# Train commands
| Command | Description|
|--------|-------|
| `framefm train run` | Starts a model training run. Configuration is loaded from the `configs/` directory using Hydra. You can override any config value directly from the command line without editing any YAML files. |
| `framefm train run [OPTIONS] [OVERRIDES]...` | Starts a model training run with the options and overrides to hydra config.|


> ### Options
> | Option | Short | Description |
> |--------|-------|-------------|
> | `--verbose` | `-v` | Print the fully resolved Hydra config to screen before training starts. |
> |` --scheduler` | `-s`| The TorchX scheduler to use for running the training job.'local_cwd' runs immediately, others submit jobs.Default is local_cwd
> | `--help` | | Show help and exit. |

> ### Overrides
> Any positional arguments after `train run` are passed to Hydra as config overrides.
> <div style="border: 2px solid #444; border-radius: 8px; display: flex; justify-content: flex-start;">

> | Syntax | Meaning | Example |
> |--------|---------|---------|
> | `key=value` | Override an existing value. Raises an error if the key does not exist. | `seed=99` |
> | `+key=value` | Append a new key. Raises an error if the key already exists. | `+experiment=baseline` |
> | `++key=value` | Override or append — safe either way. | `++new_key=99` |
> | `~key=value` | Remove a key from the config. | `~logging=demo_mlflow` |
> </div>

## Examples of the train run command with options and overrides

## FrameFM Train Commands
<div style="border: 2px solid #444; padding: 10px; border-radius: 8px;">

| Use Case | Command | Description |
| :--- | :--- | :--- |
| **Run with defaults** | `framefm train run` | Executes the training pipeline using the default configuration settings. |
| **Override config group** | `framefm train run model=demo_autoencoder` | Replaces a top-level config group (like `model` or `data`) with a specific choice. |
| **Override multiple** | `framefm train run model=demo_autoencoder data=land_cover_map seed=99` | Combines several overrides in a single execution. |
| **Override nested** | `framefm train run model=demo_autoencoder model.lr=1e-4` | Targets a specific parameter within a config group (e.g., learning rate). |
| **Append experiment** | `framefm train run +experiment=baseline` | Uses the `+` prefix to add a new experiment configuration file to the run. |
| **Remove logging** | `framefm train run ~logging=demo_mlflow` | Uses the `~` prefix to disable or remove a specific logging configuration. |
| **Print resolved config** | `framefm train run --verbose model=demo_autoencoder` | Uses the `--verbose` flag to display the final configuration before training starts. |
</div>

---
# Running FRAME-FM Training on Slurm
When using SLURM please ensure you have activate the env before scheduling the training.

## Command to start a job

Run your training via TorchX:

```bash
framefm train run model=demo_autoencoder
```

**Where to check the Slurm wrapper logs**

TorchX generates a wrapper log when submitting via Slurm:
if no job_dir was specified in config then(look under the platforms config.) 
/home/users/<username>/FRAME-FM/slurm-<jobid>.out

**Where to check the actual training logs**
The real training output is written by the worker:

/home/users/<username>/FRAME-FM/slurm-<jobid>-worker-0.out
/home/users/<username>/FRAME-FM/slurm-<jobid>-worker-0.err

worker-0.out → stdout from your training process (metrics, progress).
worker-0.err → stderr (errors, exceptions).

---
# Useful Slurm Commands for FRAME-FM Jobs

| Command | Description / Use | Notes / Examples |
|---------|-----------------|----------------|
| `squeue -u $USER` | Lists all current jobs submitted by your user | Shows job ID, name, state, partition, nodes, runtime, etc. Useful to quickly check running or pending jobs. |
| `sacct -u $USER -S <start_date> --format=JobID,JobName,State,Partition,Start,End` | Shows historical jobs for your user starting from `<start_date>` | Example: `sacct -u $USER -S 2026-03-24 --format=JobID,JobName,State,Partition,Start,End` lists all jobs submitted today with their start/end times. |
| `scontrol show job <JobID>` | Shows detailed info for a specific job | Example: `scontrol show job 10176547` displays stdout/stderr paths, resources, nodes, time limits, QoS, etc. |
| `tail -f <stdout_file>` | Follows real-time stdout logs of a job | Example: `tail -f /home/users/<user>/FRAME-FM/slurm-10176547-worker-0.out` shows the actual training logs. |
| `scancel <JobID>` | Cancels a specific job | Example: `scancel 10176547` stops that job. |
| `scancel -u $USER` | Cancels all jobs for your user | Useful to clear all pending/running jobs before starting new experiments. |
| `sacct -j <JobID> --format=JobID,JobName,State,ExitCode,Elapsed` | Check the status and exit code of a specific job | Example: `sacct -j 10176547 --format=JobID,JobName,State,ExitCode,Elapsed` helps debug why a job failed. |
| `squeue -j <JobID>` | Check if a specific job is still running/pending | Example: `squeue -j 10176547` |
| `scontrol requeue <JobID>` | Requeues a failed or cancelled job | Useful when a job hits a transient error and you want to retry without resubmitting. |

**Tips:**  
- `StdOut` and `StdErr` from `scontrol show job <JobID>` show the exact log file locations.  
- `squeue` only shows **active/pending jobs**, while `sacct` can show **completed, failed, or canceled jobs**.  
- Use `tail -f` on the worker logs to monitor training in real time.  
---

# Configuration
## Hydra Config
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

## Running with Python

If you would like to run FRAME-FM without using the CLI then you can run `train.py` directly with Python.

```bash
python src/FRAME_FM/training/train.py
```

---

## Technical Documentation

FRAME-FM uses Sphinx to generate and store technical documentation. This is a mixture of hand-written
and autogenerated documentation. For example, thorough details on the configs and transformers.

To view this documentation, visit https://nerc-eds.github.io/FRAME-FM/.


## Further reading

- [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)
- [Hydra config groups](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
