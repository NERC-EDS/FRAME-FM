# Planning Document: train_gp_spatiotemporal.py

## Overview
Create a new example file `train_gp_spatiotemporal.py` that demonstrates training a Gaussian Process model on spatiotemporal oceanographic data using the full stack: GPyTorch, PyTorch Lightning, Hydra, MLflow, and TorchX.

## Objectives
- Demonstrate GP modeling on real-world oceanographic data (Bella glider dataset)
- Show integration of GPyTorch with PyTorch Lightning
- Maintain consistency with existing example files
- Provide working code for spatiotemporal prediction with uncertainty quantification

## Data Source
**Dataset**: Bella_626_R glider data from CSV file in `data/` folder

**Direct CSV Loading**: Use the CSV file directly (not Croissant/TFDS) for simplicity in this example.

**Input Features (X)**:
- Spatial: Latitude, Longitude
- Temporal: Time (converted to days since first observation)
- Additional: Depth (GLIDER_DEPTH)

**Target Variable (Y)**:
- Temperature (TEMP) - regression task

**Rationale**: This creates a 4D spatiotemporal problem (lat, lon, time, depth → temperature), which is perfect for demonstrating GP capabilities with ARD (Automatic Relevance Determination).

## Architecture Components

### 1. Data Module: `GliderDataModule`
Inherits from `pl.LightningDataModule`

**Responsibilities**:
- Load Bella glider CSV data
- Handle missing values (NaN imputation or filtering)
- Normalize/standardize features
- Create train/validation splits
- Convert to PyTorch tensors
- Provide dataloaders

**Key Methods**:
- `__init__(csv_path, batch_size, val_split, num_workers)`
- `setup(stage)` - load and split data
- `train_dataloader()` - return training dataloader
- `val_dataloader()` - return validation dataloader

### 2. Model: `GPSpatioTemporalModel`
Use the corrected version from `gp_spatiotemporal_model_corrected.py`

**Modifications needed**:
- Adapt input dimensions for glider data (4D: lat, lon, time, depth)
- Configure appropriate kernel (RBF + Matern for spatial, Periodic for time)
- Set number of inducing points (start with 100-200)

### 3. Configuration: `conf/config_gp.yaml`
Following Hydra composition pattern

**Structure**:
```yaml
defaults:
  - config  # inherit base config

data:
  csv_path: "data/Bella_626_R_*.csv"  # Path to CSV file
  batch_size: 32  # Smaller batch for GP memory efficiency
  val_split: 0.2  # Temporal split: first 80% train, last 20% val
  num_workers: 0
  
model:
  input_dim: 4  # lat, lon, time, depth
  output_dim: 1  # temperature
  num_inducing: 150
  learning_rate: 0.01
  kernel_type: "rbf_ard"  # RBF with ARD for all 4 dimensions
  ard_num_dims: 4

mlflow:
  experiment_name: "gp_spatiotemporal_glider"
  run_name: "gp_glider_run"

trainer:
  max_epochs: 50  # GPs may need more epochs
```

### 4. Main Training Script: `train_gp_spatiotemporal.py`

**Structure** (following the pattern of other examples):

```
Imports
├── PyTorch & PyTorch Lightning
├── GPyTorch
├── Hydra & OmegaConf
├── MLflow
└── Standard libraries (pandas, numpy, etc.)

Data Module Class
├── __init__
├── setup
├── train_dataloader
└── val_dataloader

Model Class (imported from corrected file)

Main Function (@hydra.main)
├── Seed everything
├── Setup MLflow
├── Initialize data module
├── Initialize model
├── Initialize trainer with Lightning callbacks
└── Fit model

if __name__ == "__main__"
```

## Key Implementation Details

### Data Preprocessing
1. **Load CSV data** using pandas
2. **Select relevant columns**: latitude, longitude, time, GLIDER_DEPTH, TEMP
3. **Handle time**: Convert ISO datetime to days since first observation
   ```python
   # Convert timestamp to days since start
   min_timestamp = df['time'].min()
   df['time_days'] = (df['time'] - min_timestamp).dt.total_seconds() / (24 * 3600)
   # Store min_timestamp for later inverse transform
   ```
4. **Remove NaN values**: Drop rows with NaN in key columns (lat, lon, time, depth, temp)
5. **Sort by time**: Critical for temporal split
   ```python
   df = df.sort_values('time')
   ```
6. **Train/Val split**: **Always use temporal split** (first 80% train, last 20% val)
   ```python
   split_idx = int(len(df) * 0.8)
   train_df = df.iloc[:split_idx]
   val_df = df.iloc[split_idx:]
   ```
7. **Normalize features**: Fit StandardScaler on **train only**, then transform both
   ```python
   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_val_scaled = scaler.transform(X_val)  # Use train statistics
   # Save scaler for inference
   ```

### Kernel Configuration
For spatiotemporal oceanographic data, use a single RBF kernel with ARD:
```python
# Single RBF kernel with separate lengthscales for each dimension
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(ard_num_dims=4)  # lat, lon, time, depth
)
```

**Rationale**: 
- ARD (Automatic Relevance Determination) allows the model to learn different lengthscales per dimension
- Simpler than separable space-time kernels for this example
- Configurable via Hydra (can experiment with other kernels later)

### Inducing Points Initialization
- **Use random subset** of training data for simplicity
- Sample uniformly from training data
- Start with 100-150 points (balance between accuracy and speed)
```python
# Random subset from training data
random_indices = torch.randperm(len(X_train))[:num_inducing]
inducing_points = X_train[random_indices]
```

### Training Strategy
- Use Adam optimizer
- Learning rate: 0.01 (typical for GP training)
- May need 20-50 epochs for convergence
- Monitor validation ELBO
- Early stopping on validation loss

### MLflow Logging
- Hyperparameters: num_inducing, learning_rate, kernel_type
- Metrics: train_loss, val_loss (ELBO)
- Artifacts: final model checkpoint
- Log GP kernel hyperparameters (lengthscales, noise)

## Expected Challenges & Solutions

### Challenge 1: Large Dataset
**Issue**: Glider dataset may have 10,000+ points, expensive for GP
**Solution**: Use variational sparse GP with inducing points (already planned)

### Challenge 2: Different Feature Scales
**Issue**: Lat/lon in degrees, time in days, depth in meters - all different scales
**Solution**: Standardize all features to zero mean, unit variance
- **Critical**: Fit scaler on train data only, then transform validation
- Prevents data leakage

### Challenge 3: Temporal Encoding
**Issue**: Time as ISO string needs conversion, raw timestamps are too large
**Solution**: Convert to days since first observation
```python
time_days = (timestamp - min_timestamp) / (24 * 3600)  # days since start
```
- Keeps numbers reasonable (e.g., 0-100 days instead of millions of seconds)
- Store `min_timestamp` for inverse transform during inference

### Challenge 4: Memory Usage
**Issue**: GP covariance matrices can be large (O(N²) memory, O(N³) computation)
**Solution**: 
- Use smaller batch sizes (**16-32**, not 64)
- Limit inducing points (100-150)
- Use float32 instead of float64
- Variational inference helps but doesn't eliminate the issue

### Challenge 5: Validation Metrics
**Issue**: ELBO loss is not interpretable to domain scientists
**Solution**: Add MAE and RMSE in temperature units (°C)
```python
def validation_step(self, batch, batch_idx):
    # Compute ELBO loss
    # ...
    
    # Also compute interpretable metrics
    with torch.no_grad():
        pred = self.likelihood(self(x)).mean
        mae = torch.abs(pred - y).mean()
        rmse = torch.sqrt(((pred - y) ** 2).mean())
    
    self.log('val_mae', mae)
    self.log('val_rmse', rmse)
```

## Testing Strategy

### Unit Testing
- Test data loading with small CSV subset
- Test feature normalization
- Test model forward pass

### Integration Testing
- Run 1-2 epochs on small data subset
- Verify MLflow logging works
- Check TorchX compatibility (can add component later)

### Validation
- Check that validation loss decreases
- Verify uncertainty estimates are reasonable
- Plot predictions vs actuals (manual verification)

## File Structure

```
frame_project/
├── train_gp_spatiotemporal.py          # New main script
├── conf/
│   └── config_gp.yaml                   # New config
├── data/
│   └── Bella_626_R_*.csv               # Data file (from JSON-LD)
├── gp_spatiotemporal_model_corrected.py # Existing (may import from here)
└── notes/
    └── gp_example_plan.md              # This file
```

## Additional Features

### Prediction Method
Add a `predict_step` method for inference:
```python
def predict_step(self, batch, batch_idx):
    x, _ = batch
    self.eval()
    self.likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = self.likelihood(self(x))
        mean = pred_dist.mean
        variance = pred_dist.variance
    
    return {'mean': mean, 'variance': variance}
```

### Kernel Hyperparameter Logging
Log learned lengthscales after training:
```python
# After training (in main or on_train_end callback)
self.log('lengthscale_lat', self.covar_module.base_kernel.lengthscale[0, 0])
self.log('lengthscale_lon', self.covar_module.base_kernel.lengthscale[0, 1])
self.log('lengthscale_time', self.covar_module.base_kernel.lengthscale[0, 2])
self.log('lengthscale_depth', self.covar_module.base_kernel.lengthscale[0, 3])
self.log('outputscale', self.covar_module.outputscale)
self.log('noise', self.likelihood.noise)
```

### Prediction Demo
Add a simple demo at the end of training:
```python
# After trainer.fit()
print("\n=== Prediction Demo ===")
val_batch = next(iter(datamodule.val_dataloader()))
predictions = trainer.predict(model, dataloaders=[val_batch])
mean = predictions[0]['mean']
std = predictions[0]['variance'].sqrt()

print(f"Sample predictions (mean ± std):")
for i in range(min(5, len(mean))):
    print(f"  {mean[i]:.2f} ± {std[i]:.2f} °C")
```

## Success Criteria

1. ✅ Script runs without errors
2. ✅ Model trains and loss decreases
3. ✅ MLflow experiment is created and logged
4. ✅ Code follows same pattern as other examples
5. ✅ Includes Google-style docstrings
6. ✅ Configuration uses Hydra composition
7. ✅ Compatible with TorchX (similar structure)
8. ✅ Logs interpretable metrics (MAE, RMSE)
9. ✅ Logs kernel hyperparameters
10. ✅ Includes prediction demo

## Next Steps

1. Create `conf/config_gp.yaml` with ARD kernel config
2. Implement `GliderDataModule` with:
   - CSV loading
   - Temporal split (sort by time, first 80% train)
   - Proper scaler fitting (train only)
   - Time encoding (days since start)
3. Adapt `GPSpatioTemporalModel` for:
   - 4D input with RBFKernel(ard_num_dims=4)
   - Interpretable metrics in validation_step
   - predict_step method
   - Kernel hyperparameter logging
4. Implement main training function with:
   - Hydra configuration
   - MLflow logging
   - Prediction demo at end
5. Test with small data subset
6. Run full training and verify results
7. Document in walkthrough

## Notes

- Keep code simple and well-commented (it's an example file)
- Add comments explaining GP-specific concepts (inducing points, ELBO, etc.)
- Make sure it aligns with the oceanographer markdown guidance
- Consider adding visualization code (optional, for debugging)
