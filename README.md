# PCA-LGBM-GP-based BO

This code implements the PCA-LightGBM-GP-based Bayesian Optimization hybrid automatic
calibration framework for SWMM proposed in the paper. The framework screens key parameters
using LHS-PRCC (cross-checked with a surrogate-based Sobol analysis), compresses multi-site,
multi-event water level time series via PCA, and compares inverse LightGBM, stand-alone
Bayesian optimization, and a LightGBM-warm-started hybrid. Each optimization strategy is
evaluated under two objective formulations: an EVR-weighted distance in the PCA domain and a
direct time-domain RMSE.

The observations are treated with a minimum-baseline correction, while the simulated series are
kept without baseline removal and stabilised through a z-score scale floor. This addresses the
base-level offset that arises because the target network is combined whereas the SWMM model is
configured as storm-only.

<img width="1535" height="591" alt="Graphical abstract_1" src="https://github.com/user-attachments/assets/697311ee-4ba4-4920-82dd-2c2c122b5417" />

## Code Structure

The calibration workflow is implemented through 12 sequential Python scripts in `METHOD_CODE/`:

| Script | Description |
|--------|-------------|
| `1.LHS` | Generate parameter samples and create event-specific SWMM input files (10-parameter screening design) |
| `2.PYSWMM` | Execute SWMM simulations for the sampled parameter sets |
| `3.PRCC` | Compute rank-based partial correlation (PRCC) sensitivity indices and identify key calibration parameters |
| `4.Sobol_surrogate` | Surrogate (GP)-based Sobol first-order and total-order indices, as a robustness cross-check of the PRCC screening |
| `5.PCA` | Resample simulated water levels to a fixed length (no baseline removal), apply a floored z-score, fit PCA to 95% cumulative variance, and export PC scores, per-event Y matrices, and EVR weights |
| `6.PCA_OBS` | Project minimum-baseline corrected observations into the fitted PCA space |
| `7.CASE_A_LGBM` | Case A: train inverse LightGBM models mapping PC scores to parameters |
| `8.CASE_B1_BO_EVR` | Case B-1: stand-alone GP-BO using the EVR-weighted PCA-domain objective |
| `9.CASE_B2_BO_RMSE` | Case B-2: stand-alone GP-BO using the direct time-domain RMSE objective |
| `10.CASE_C1_HYBRID_EVR` | Case C-1: hybrid with LightGBM warm-start and the EVR-weighted objective |
| `11.CASE_C2_HYBRID_RMSE` | Case C-2: hybrid with LightGBM warm-start and the direct RMSE objective |
| `12.Apply and Run PYSWMM` | Apply optimal parameters and run validation simulations, including the independent validation event |

Cases B-1/B-2 share an identical initial point, bounds, evaluation budget, and random seed, and
differ only in the objective function; the same holds for C-1/C-2. This isolates the effect of
the objective formulation. Calibration uses events R1, R2, R5, and R6; R8 is reserved as an
independent validation event.

## Dependencies

`pyswmm`, `scikit-learn`, `lightgbm`, `SALib`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
