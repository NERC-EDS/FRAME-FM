# Observability
## Overview

FRAME-FM integrates with [MLflow](https://www.mlflow.org/docs/latest/ml/) to provide experiment observability for
all model‑training workflows within the framework. The MLflow UI can be used to compare metrics between model variants or datasets.

MLflow support is implemented via a custom MLflow logger that captures metrics, parameters, and metadata about the training run.

By default, the FRAME‑FM's main configuration file ([config.yaml](../../../configs/config.yaml)) references a logging configuration
([demo_mlflow.yaml](../../../configs/logging/demo_mlflow.yaml)) located in configs/logging/. This file demonstrates how to:

- configure an MLflow logger instance
- set experiment names
- define run names
- attach user-defined tags (e.g., project, dataset, model)

These settings allow you to organise training runs and compare results across models, datasets, and configurations.

## MLflow server

### Prerequisites

MLflow is included in the FRAME‑FM environment. After following the environment setup steps in the main [README](../../../README.md), MLflow will be available automatically. 

If you are working outside the FRAME‑FM environment, you can install MLflow manually:
```shell
pip install mlflow
```

### Running a local MLflow server

For local development, you can start a lightweight MLflow tracking server and UI.

Inside the FRAME‑FM environment (using uv):
```shell
uv run mlflow server --port 5000
```

Outside FRAME‑FM (using a regular Python environment):
```shell
mlflow server --port 5000
```

Once running, the MLflow UI is available at:

http://127.0.0.1:5000

This interface allows you to track experiment histories, compare metrics, and review metadata for reproducibility.

## Logging Configuration

MLflow logging in FRAME‑FM is controlled by Hydra configuration files stored in `configs/logging/`.
Below is an example configuration ([demo_mlflow.yaml](../../../configs/logging/demo_mlflow.yaml)):

```yaml
_target_: FRAME_FM.training.logger.create_mlflow_logger

experiment_name: "frame-fm-mmmae-demo"
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI, "sqlite:///mlflow.db"}
run_name: "initial-demo"

tags:
    project: "FRAME-FM"
    dataset: "Spatial Land Cover"
    model: "Multimodal Masked Autoencoder"
```

### Key Fields

#### `_target_`
The function responsible for instantiating the MLflow logger.

#### `experiment_name`
Groups related training runs under an MLflow experiment.

#### `tracking_uri`
Points to the MLflow tracking backend. FRAME‑FM defaults to a local SQLite file (mlflow.db) unless overridden via the `MLFLOW_TRACKING_URI` environment variable.

#### `run_name`
Human-readable name for the run, visible in the MLflow UI.

#### `tags`
Custom metadata for filtering or searching runs. Useful tags include:
- Project identifiers
- Dataset names
- Model architectures
- Training configurations

## Storage locations for metadata and artifacts

MLflow uses two storage components:

- **Backend store** for experiment metadata, params, metrics
- **Artifact store** for files generated during runs (models, plots, logs, checkpoints)

FRAME-FM's example MLflow configuration ([demo_mlflow.yaml](../../../configs/logging/demo_mlflow.yaml)) sets the backend store location via the `MLFLOW_TRACKING_URI` environment variable. If this is not set, this configuration defaults to storing metadata in a SQLite database using an `mlflow.db` database file located in the project root.

The artifact store location defaults to `./mlruns/` (but artifacts are not currently stored when using FRAME-FM).

MLflow will create the directory structure automatically if it doesn't exist.

When starting an MLflow server, you must point it to the same backend store and artifact directory used by FRAME‑FM in order to view your runs in the MLflow UI. If you start the server with different paths, MLflow will initialise new empty stores and your existing runs will not appear in the UI.

You can specify storage paths explicitly when launching the server via `--backend-store-uri` and `--default-artifact-root`:

```shell
uv run mlflow server \
  --backend-store-uri sqlite:///path/to/mlflow.db \
  --default-artifact-root file:/path/to/mlruns \
  --host 127.0.0.1 \
  --port 5000
```
or, if you prefer using the `MLFLOW_TRACKING_URI` environment variable:
```shell
export MLFLOW_TRACKING_URI="sqlite:///path/to/mlflow.db"
uv run mlflow server \
  --default-artifact-root file:/path/to/mlruns \
  --host 127.0.0.1 \
  --port 5000
```

## Running MLflow on JASMIN and viewing locally

You can run the MLflow tracking server on a JASMIN host and view the UI from your local machine via SSH port forwarding.

First, start an MLflow server on JASMIN (e.g., a `sci` node), ensuring `backend-store-uri` points to the configured MLflow tracking backend.

```shell
uv run mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root file:./mlruns \
--host 127.0.0.1 \
--port 5000
```

To access the MLflow UI in your local browser:
Access the UI from your local browser by forwarding your local port to the JASMIN host, e.g., if running FRAME-FM on the `sci` servers:
1. Ensure your JASMIN SSH access is configured correctly.
    See [JASMIN documentation](https://help.jasmin.ac.uk/docs/interactive-computing/login-servers/#connecting-to-a-sci-server-via-a-login-server) 
   for more information on connecting through the login servers to a `sci` node.
2. Forward your local port 5000 to port 5000 on the `sci` node (adjust hostname as needed):
    ```shell
    ssh -L 5000:localhost:5000 -J <USERNAME>@login.jasmin.ac.uk <USERNAME>@sci-vm-04.jasmin.ac.uk
    ```
3. Open your local browser and navigate to: http://127.0.0.1:5000

You should now see the MLflow UI for the runs created on the JASMIN machine.

![Screenshot of MLflow experiment overview page](images/mlflow_overview.png)
![Screenshot of Mlflow experiment metrics page](images/mlflow_metrics.png)
