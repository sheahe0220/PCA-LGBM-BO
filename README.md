# PCA-LGBM-BO
Implementation of a hybrid automatic calibration framework for SWMM combining Principal Component Analysis, LightGBM, and Bayesian Optimization.
### 1. `1_lhs_sampling.py`
- Generates Latin Hypercube Samples (N=300) for 10 SWMM parameters
- Creates event-specific SWMM input files

### 2. `2_run_pyswmm.py`
- Executes SWMM simulations for all LHS samples in parallel
- Processes multiple rainfall events (R1, R2, R5, R6)

### 3. `3_prcc_analysis.py`
- Performs Partial Rank Correlation Coefficient (PRCC) sensitivity analysis
- Identifies 4 key parameters: Imperv_scale, Width_scale, Nimp_scale, n_pipe_scale

### 4. `4_resample_depth.py`
- Resamples water level time series to fixed length (N=480)
- Applies baseline removal and normalization

### 5. `5_pca_training.py`
- Trains PCA models on simulated water levels (95% explained variance)
- Performs Z-score standardization and dimensionality reduction

### 6. `6_pca_obs_transform.py`
- Transforms observed water level data using trained PCA models
- Projects observations onto principal component space

### 7. `7_case_a_lgbm.py`
- **Case A**: Inverse LightGBM mapping (PCA scores → parameters)
- Event-wise prediction with θ-space pooling

### 8. `8_case_b_bo.py`
- **Case B**: Direct Bayesian Optimization with random initialization
- Gaussian Process-based parameter search

### 9. `9_case_c_hybrid.py`
- **Case C**: Hybrid approach (LGBM initialization + BO refinement)
- Warm-start BO from Case A results

### 10. `10_apply_and_run.py`
- Applies optimal parameters to SWMM model
- Runs validation simulations
