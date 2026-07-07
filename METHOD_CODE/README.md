# METHOD_CODE_revised

This folder contains the calibration pipeline as reported in the **accepted (revised) manuscript**
(Journal of Hydroinformatics, HYDRO-D-26-00025). It supersedes the original `METHOD_CODE/`
and reflects three methodological updates introduced during revision:

1. **Baseline handling.** The earlier first-10%-median baseline removal is replaced by a
   *minimum-baseline* correction on the observations, while the simulated series are kept
   without baseline removal and stabilised through a z-score *scale floor* (`scale_floor = 0.01`,
   at the sensor precision of about 1 cm). This treatment addresses the dry-weather-flow and
   base-level offset that arises because the target network is combined but the SWMM model is
   configured as storm-only.
2. **Objective-function comparison.** The calibration objective is examined in two forms:
   an EVR-weighted distance in the PCA domain, and a direct time-domain RMSE. Each is applied
   to both the stand-alone Bayesian optimization and the LightGBM-warm-started hybrid.
3. **Four calibration strategies.** Inverse LightGBM (Case A), stand-alone GP-BO (Cases B-1/B-2),
   and the LightGBM-warm-started hybrid (Cases C-1/C-2) are compared under a common budget and seed.

A surrogate-based Sobol analysis is also included as a robustness cross-check of the
LHS-PRCC screening.

## Code structure

| Script | Status vs `METHOD_CODE/` | Description |
|--------|--------------------------|-------------|
| `1.LHS` | unchanged | Generate LHS parameter samples and event-specific SWMM input files (10-parameter screening design). |
| `2.PYSWMM` | unchanged | Execute SWMM simulations for the sampled parameter sets. |
| `3.PRCC` | unchanged | Compute rank-based partial correlation (PRCC) sensitivity indices and identify key calibration parameters. |
| `4.Sobol_surrogate` | added | Surrogate (GP)-based Sobol first-order and total-order indices as a robustness cross-check of PRCC. |
| `5.PCA` | changed | Resample simulated water levels to a fixed length (no baseline removal), apply a floored z-score, fit PCA to 95% cumulative variance, and export PC scores, per-event `Y_{event}_selected.csv`, and EVR weights. |
| `6.PCA_OBS` | changed | Project minimum-baseline corrected observations into the fitted PCA space. |
| `7.CASE_A_LGBM` | changed | Case A: inverse LightGBM mapping from PC scores to parameters (pooled and event-wise). |
| `8.CASE_B1_BO_EVR` | changed | Case B-1: GP-BO from a random init using the EVR-weighted PC-domain objective. |
| `9.CASE_B2_BO_RMSE` | added | Case B-2: GP-BO from the same random init using a direct time-domain RMSE objective. |
| `10.CASE_C1_HYBRID_EVR` | changed | Case C-1: hybrid with LightGBM warm-start and the EVR-weighted objective. |
| `11.CASE_C2_HYBRID_RMSE` | added | Case C-2: hybrid with LightGBM warm-start and the direct RMSE objective. |
| `12.Apply and Run PYSWMM` | unchanged | Apply optimal parameters and run validation simulations, including the independent validation event (R8). |

Cases B-1/B-2 share an identical initial point, bounds, budget, and seed, and differ only in the
objective; the same holds for C-1/C-2. This isolates the effect of the objective function.

## Calibration events

- Calibration: R1, R2, R5, R6.
- Independent validation: R8 (excluded from PCA fitting and BO; used only in `12.Apply and Run PYSWMM`).

## Expected data layout

All scripts read paths from a `USER CONFIG` block at the top of the file. The default layout,
rooted at `BASE`, is:

```
BASE/
  SWMM/                         event templates: 10mm_{event}.inp
  LHS_R/                        LHS samples (lhs_samples_10d.csv) and per-event INP files
  LHS_RESULTS/                  raw SWMM output from 2.PYSWMM
  LHS_DEPTH/{event}/{node}/     per-sample simulated depth series (input to 5.PCA)
  LHS_OUT_RESULTS/{event}/      per-node summary statistics (input to 3.PRCC, 4.Sobol)
  PRCC/                         PRCC results
  input/
    lhs_samples_4d.csv          post-screening 4-parameter samples
    obs_depth/                  raw observed depth
  OBS_min_baseline/{event}/     minimum-baseline corrected observations (Depth_adjusted)
  PCA/
    models/{node}/              scaler.pkl, pca.pkl, explained_variance.csv
    scores/{event}/{node}/      PCA_scores_*.csv
    obs_pc/                     observed PC scores and obs_pc_summary.csv
    Y_{event}_selected.csv      merged model PC scores per event
    PC_selected_evr.csv         EVR weights for the objective function
  RESULTS/
    Sobol/                      Sobol indices and figures
    CaseA_LGBM/                 inverse LightGBM models and pooled theta
    CaseB1_EVR/, CaseB2_RMSE/   BO run histories and best results
    CaseC1_HYBRID_EVR/, CaseC2_HYBRID_RMSE/   hybrid run histories and best results
```

## Dependencies

`pyswmm`, `scikit-learn`, `lightgbm`, `SALib`, `scipy`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
