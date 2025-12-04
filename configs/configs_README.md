# configs/

Hydra configuration files defining models, data modules, trainers, logging, and experiment settings.  
All training and evaluation behaviour is controlled via these YAML configs.

## How it works
The top-level `config.yml` defines the default configuration by including entries from the subfolders:
```yaml
defaults:
  - model: simple
  - data: dummy
  - trainer: default
  - logging: mlflow
  - experiment: base
```

Hydra then loads the corresponding files:

- `configs/model/simple.yml`
- `configs/data/dummy.yml`
- `configs/trainer/default.yml`
- `configs/logging/mlflow.yml`
- `configs/experiment/base.yml`

These files are merged together into one final config object, which is passed into the training script.

## Subdirectories:
- `model/`

Defines model architectures and their hyperparameters.
Each file here must contain a _target_ field that points to a Python class implementing the model.
Example: selecting different neural network variants or LightningModules.

- `data/`

Configures DataModules, dataloaders, dataset paths, and batch sizes.
Each file should describe how training/validation/test data is created or loaded.

- `trainer/`

Contains configurations for PyTorch Lightning’s Trainer:
number of epochs, devices, precision settings, logging frequency, etc.

- `logging/`

Parameters for MLflow or other logging backends.
This keeps experiment tracking separate from model or data settings.

- `experiment/`

Optional presets for organising runs (name, notes, parameter overrides).
Useful for keeping track of different experiment configurations.

### Using the configs
Hydra is invoked in `train.py` via:
```python
@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    ...
```
You can start training with the default config:
```bash
python -m FRAME_FM.training.train.py
```
Or override specific config options from the command line:
```bash
python -m FRAME_FM.training.train.py model=simple data=cifar10 trainer.max_epochs=50
```
