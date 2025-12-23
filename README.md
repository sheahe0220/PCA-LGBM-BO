# PCA-LGBM-GP-bades BO
This code implements the PCA–LightGBM–GP-based Bayesian Optimization hybrid automatic calibration framework for SWMM proposed in the paper. The framework screens key parameters using LHS–PRCC, compresses multi-site, multi-event water level time series via PCA, and compares three calibration strategies: Inverse LGBM, BO, and Hybrid.

<img width="1535" height="591" alt="Graphical abstract_1" src="https://github.com/user-attachments/assets/697311ee-4ba4-4920-82dd-2c2c122b5417" />

## Code Structure

The calibration workflow is implemented through 10 sequential Python scripts in `METHOD_CODE/`:

| Script | Description |
|--------|-------------|
| `1.LHS` | Generate parameter samples and create event-specific SWMM input files |
| `2.PYSWMM` | Execute SWMM simulations for the sampled parameter sets |
| `3.PRCC` | Compute sensitivity indices and identify key calibration parameters |
| `4.Resample` | Align water-level time series by resampling to a fixed length and removing the initial baseline offset |
| `5.PCA` | Train PCA on simulated water levels and extract principal component representations |
| `6.PCA_OBS` | Project observed water-level series into the PCA space |
| `7.CASE_A` | Train inverse mapping models from PC scores to parameters |
| `8.CASE_B` | Perform direct Bayesian optimization using SWMM simulations as the forward model |
| `9.CASE_C` | Warm-start BO with inverse LightGBM estimates and refine via optimization |
| `10.Apply and Run PYSWMM` | Apply optimal parameters and run validation simulations |
