# Code Review - Oceanographer ML Framework Markdown

## Errors Found in Code Examples

### 1. GPSpatioTemporalModel - ApproximateGP Initialization (Lines 77-86)

**Error**: The `ApproximateGP` constructor is being called incorrectly.

**Current (Incorrect)**:
```python
self.gp_layer = gpytorch.models.ApproximateGP(
    gpytorch.variational.VariationalStrategy(
        self,
        inducing_points,
        gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        ),
        learn_inducing_locations=True
    )
)
```

**Issue**: `ApproximateGP` is a base class that should be inherited, not instantiated as a layer. The `VariationalStrategy` should be passed to the base class constructor.

**Correct Approach**:
```python
class GPSpatioTemporalModel(gpytorch.models.ApproximateGP, pl.LightningModule):
    def __init__(self, inducing_points, feature_extractor=None):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        gpytorch.models.ApproximateGP.__init__(self, variational_strategy)
        pl.LightningModule.__init__(self)
        
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=3) *
            gpytorch.kernels.PeriodicKernel()
        )
```

### 2. GPSpatioTemporalModel - Missing mean_module (Line 99)

**Error**: The code calls `self.gp_layer.mean_module(x)` but `mean_module` is never defined.

**Fix**: Add `self.mean_module` in `__init__` and use it in forward.

### 3. GPSpatioTemporalModel - validation_step (Lines 115-121)

**Error**: Using `likelihood.log_marginal()` which doesn't exist in this context.

**Current (Incorrect)**:
```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    with gpytorch.settings.num_likelihood_samples(32):
        output = self(x)
        loss = -self.likelihood.log_marginal(y, output)
    self.log('val_loss', loss)
    return loss
```

**Issue**: `GaussianLikelihood` doesn't have a `log_marginal()` method. For validation, you should use the same ELBO or use predictive log probability.

**Correct Approach**:
```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    output = self(x)
    
    # Use the same ELBO for validation
    mll = gpytorch.mlls.VariationalELBO(
        self.likelihood, self, num_data=y.size(0)
    )
    loss = -mll(output, y)
    
    self.log('val_loss', loss)
    return loss
```

### 4. VariationalELBO - Incorrect instantiation (Lines 108-110)

**Error**: Passing `self.gp_layer` to `VariationalELBO` when it should be `self`.

**Current (Incorrect)**:
```python
loss = -gpytorch.mlls.VariationalELBO(
    self.likelihood, self.gp_layer, num_data=len(self.trainer.train_dataloader.dataset)
)(output, y)
```

**Correct**:
```python
mll = gpytorch.mlls.VariationalELBO(
    self.likelihood, self, num_data=len(self.trainer.train_dataloader.dataset)
)
loss = -mll(output, y)
```

### 5. Hydra instantiate - Incorrect usage (Line 257)

**Error**: Trying to instantiate a string model name.

**Current (Incorrect)**:
```python
model = GPSpatioTemporalModel(
    inducing_points=cfg.model.inducing_points,
    feature_extractor=hydra.utils.instantiate(cfg.model.backbone)
)
```

**Issue**: `cfg.model.backbone` is just `"resnet18"` (a string), not a Hydra config object. You can't instantiate a string.

**Correct Approach**:
```python
# Option 1: Don't use instantiate for simple model names
import torchvision.models as models
feature_extractor = getattr(models, cfg.model.backbone)(pretrained=True)

# Option 2: Make the config proper for instantiate
# In config: backbone: {_target_: torchvision.models.resnet18, pretrained: true}
feature_extractor = hydra.utils.instantiate(cfg.model.backbone)
```

### 6. MLFlowLogger - Incorrect parameter name (Line 261)

**Error**: Using non-existent parameter in `hydra.utils.instantiate`.

**Current**: The code suggests instantiating logger via Hydra config, but the config file doesn't show a proper logger config.

**Recommendation**: Either define the logger config properly or instantiate directly:
```python
mlflow_logger = MLFlowLogger(
    experiment_name=cfg.mlflow.experiment_name,
    run_name=cfg.mlflow.run_name,
    tracking_uri=cfg.mlflow.tracking_uri,
)
```

## Summary

The main issues are:
1. **ApproximateGP misuse**: Should inherit from it, not instantiate it as a layer
2. **Missing mean_module**: Required component not defined
3. **Non-existent log_marginal method**: Use ELBO or predictive log probability instead
4. **Incorrect ELBO instantiation**: Pass `self` not `self.gp_layer`
5. **Hydra instantiate misuse**: Can't instantiate a string
6. **Logger config incomplete**: Needs proper definition or direct instantiation

These errors would prevent the code from running successfully.
