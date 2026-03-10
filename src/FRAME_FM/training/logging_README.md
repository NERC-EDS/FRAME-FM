# Observability
## Overview

This repository integrates with [MLflow](https://www.mlflow.org/docs/latest/ml/) to provide experiment observability for
all model‑training workflows within the framework. The MLflow UI can be used to compare metrics between model variants or datasets.

MLflow support is implemented via a custom MLflow logger that captures metrics, parameters, configuration, and artifacts generated during training.

By default, the framework’s main configuration file ([config.yaml](../../../configs/config.yaml)) references a logging configuration
([demo_mlflow.yaml](../../../configs/logging/demo_mlflow.yaml)) located in configs/logging/. This file demonstrates how to:

- Configure an MLflow logger instance
- Set experiment names
- Define run names
- Attach user-defined tags (e.g., project, dataset, model)

These settings allow you to organise training runs and compare results across models, datasets, and configurations.

## MLflow Tracking URI

MLflow uses the MLFLOW_TRACKING_URI environment variable to determine where to store experiment metadata and artifacts.

- **If `MLFLOW_TRACKING_URI` is set:** All run data is logged to the location or server specified.
- **If it is not set:** MLflow defaults to a local directory named mlruns in the project root.

For local development, you can start a lightweight MLflow UI server with:

```shell
uv run mlflow server \
--backend-store-uri file:./mlruns \
--default-artifact-root file:./mlruns \
--host 127.0.0.1 \
--port 5000
```

Once running, the MLflow UI is available at:

http://127.0.0.1:5000

This interface lets you track experiment histories, compare metrics visually, inspect model artifacts, and review metadata for reproducibility.

## Logging Configuration

MLflow logging is configured via Hydra YAML files in configs/logging/.
Below is an example configuration ([demo_mlflow.yaml](../../../configs/logging/demo_mlflow.yaml)) that defines a simple MLflow logger:

```yaml
_target_: FRAME_FM.training.logger.create_mlflow_logger

experiment_name: "frame-fm-mmmae-demo"
tracking_uri: ${oc.env:MLFLOW_TRACKING_URI, "file:./mlruns"}
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
Groups related training runs under a single MLflow experiment.

#### `tracking_uri`
Points to the MLflow tracking backend. Falls back to a local mlruns directory if no environment variable is set.

#### `run_name`
Human-readable name for the run, visible in the MLflow UI.

#### `tags`
Metadata for filtering or searching runs. Useful tags include:
- Project identifiers
- Dataset names
- Model architectures
- Training configurations

## Running MLflow on JASMIN and viewing locally

You can run the MLflow tracking server on a JASMIN host and view the UI from your local machine via SSH port forwarding.

First, start an MLflow server on JASMIN (e.g., a `sci` node), ensuring `backend-store-uri` points to the configured MLflow tracking backend.

```shell
uv run mlflow server \
--backend-store-uri file:./mlruns \
--default-artifact-root file:./mlruns \
--host 127.0.0.1 \
--port 5000
```

To access the MLflow UI in your local browser:
Access the UI from your local browser by forwarding your local port to the JASMIN host:
E.g., if running FRAME-FM on the `sci` servers:
1. Ensure your JASMIN SSH access is configured correctly.
    See [JASMIN documentation](https://help.jasmin.ac.uk/docs/interactive-computing/login-servers/#connecting-to-a-sci-server-via-a-login-server) 
   for more information on connecting through the login servers to a `sci` node.
2. Forward your local port 5000 to port 5000 on the `sci` node (adjust hostname as needed):
    ```shell
    ssh -L 5000:localhost:5000 -J <USERNAME>@login.jasmin.ac.uk <USERNAME>@sci-vm-04.jasmin.ac.uk
    ```
3. Open your local browser and navigate to: http://127.0.0.1:5000

You should now see the MLflow UI for the runs created on the JASMIN machine.
