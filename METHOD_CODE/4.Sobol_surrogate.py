# -*- coding: utf-8 -*-
"""
Surrogate-based Sobol Sensitivity Analysis (10D)
=================================================
Approach:
  1) Train GP surrogate on existing 300 LHS samples (per event/node/response)
  2) Generate Saltelli quasi-random samples via SALib
  3) Evaluate surrogate → compute first-order (S1) & total-order (ST) Sobol indices
  4) Interaction index = ST - S1

Input files:
  - LHS parameters : C:/Users/SGJEONG99/Desktop/new/LHS_R/lhs_samples_10d.csv
  - Summary stats   : C:/Users/SGJEONG99/Desktop/new/LHS_OUT_RESULTS/{EVENT}/{NODE}_summary.csv
  - PRCC (for comparison) : C:/Users/SGJEONG99/Desktop/new/PRCC/{EVENT}/{NODE}_PRCC.csv

Output files (-> RESULTS/Sobol/):
  - sobol_indices.csv          : full S1/ST/interaction per param/event/node/response
  - gp_cv_scores.csv           : GP surrogate 5-fold CV R² per event/node/response
  - sobol_S1_ST_barplot.png    : bar chart (max_depth_m, M0113 & MH0126)
  - sobol_heatmap.png          : heatmap of S1 across events
  - sobol_vs_prcc.png          : Sobol S1 vs PRCC² scatter (consistency check)
  - sobol_interaction.png      : interaction index (ST - S1) bar chart

Dependencies: SALib, scikit-learn, pandas, numpy, matplotlib, tqdm
  conda/pip install SALib scikit-learn
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
try:
    from SALib.sample import sobol as saltelli_sampler   # SALib ≥ 1.5
except ImportError:
    from SALib.sample import saltelli as saltelli_sampler
from SALib.analyze import sobol
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ===== USER CONFIG: change paths here =====
BASE        = Path(r"C:\Users\SGJEONG99\Desktop\new")
LHS_CSV     = BASE / "LHS_R" / "lhs_samples_10d.csv"
SUMMARY_DIR = BASE / "LHS_OUT_RESULTS"
PRCC_DIR    = BASE / "PRCC"
OUT_DIR     = BASE / "RESULTS" / "Sobol"

# ===== Settings =====
EVENTS   = ["R1", "R2", "R5", "R6"]
NODES    = ["M0113", "MH0126"]          # calibration nodes
RESPONSES = [
    "max_depth_m",
    "avg_depth_m",
    "t_idx_max_depth",
    "dur_depth_gt_0_5m_min",
    "tot_inflow_vol_m3",
    "max_total_inflow_m3s",
]
N_SALTELLI    = 1024   # Saltelli base size → total = N*(k+2) = 12,288
SEED          = 42
GP_RESTARTS   = 3
CV_FOLDS      = 5


# =====================================================================
def fit_gp_and_sobol(X, y, problem, X_saltelli):
    """Fit GP surrogate, cross-validate, compute Sobol indices."""
    # Scale
    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    X_sc = sx.transform(X)
    y_sc = sy.transform(y.reshape(-1, 1)).ravel()

    # GP kernel
    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-3)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=GP_RESTARTS,
        random_state=SEED, normalize_y=False
    )
    gp.fit(X_sc, y_sc)

    # 5-fold CV R²
    cv_r2 = cross_val_score(
        GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=GP_RESTARTS,
            random_state=SEED, normalize_y=False
        ),
        X_sc, y_sc, cv=CV_FOLDS, scoring="r2"
    ).mean()

    # Predict on Saltelli samples
    X_salt_sc = sx.transform(X_saltelli)
    y_pred = gp.predict(X_salt_sc)

    # Sobol analysis
    Si = sobol.analyze(problem, y_pred, calc_second_order=False)

    return Si, cv_r2


# =====================================================================
def load_prcc_data():
    """Load existing PRCC results for comparison."""
    rows = []
    for ev in EVENTS:
        for node in NODES:
            csv = PRCC_DIR / ev / f"{node}_PRCC.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df = df.dropna(subset=["PRCC"])
                rows.append(df)
    if rows:
        return pd.concat(rows, ignore_index=True)
    return None


# =====================================================================
def plot_s1_st_bar(df, out_dir):
    """Bar chart: S1 & ST for max_depth_m at calibration nodes."""
    df_depth = df[df["response"] == "max_depth_m"].copy()
    if df_depth.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, node in zip(axes, NODES):
        sub = df_depth[df_depth["node"] == node]
        if sub.empty:
            continue

        # Average across events
        agg = sub.groupby("param").agg(
            S1_mean=("S1", "mean"), ST_mean=("ST", "mean"),
            S1_std=("S1", "std"),   ST_std=("ST", "std")
        ).reindex(sub["param"].unique())

        # Sort by ST descending
        agg = agg.sort_values("ST_mean", ascending=False)

        x = np.arange(len(agg))
        w = 0.35
        ax.bar(x - w/2, agg["S1_mean"], w, yerr=agg["S1_std"],
               label="S1 (first-order)", color="#4C72B0", capsize=3, alpha=0.85)
        ax.bar(x + w/2, agg["ST_mean"], w, yerr=agg["ST_std"],
               label="ST (total-order)", color="#DD8452", capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"{node} — max_depth_m", fontsize=12)
        ax.set_ylabel("Sobol Index")
        ax.legend(fontsize=9)
        ax.axhline(0.05, ls="--", color="gray", lw=0.8, alpha=0.6)
        ax.set_ylim(bottom=-0.05)

    fig.suptitle("Sobol Sensitivity — max_depth_m (mean ± std across 4 events)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "sobol_S1_ST_barplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df, out_dir):
    """Heatmap: S1 for max_depth_m across events × params."""
    df_depth = df[df["response"] == "max_depth_m"].copy()
    if df_depth.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, node in zip(axes, NODES):
        sub = df_depth[df_depth["node"] == node]
        if sub.empty:
            continue

        pivot = sub.pivot_table(index="param", columns="event", values="S1")
        # Sort by mean S1
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(f"{node} — S1 (max_depth_m)", fontsize=12)

        # Annotate
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                color = "white" if v > 0.5 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_dir / "sobol_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interaction(df, out_dir):
    """Bar chart: interaction index (ST - S1) for max_depth_m."""
    df_depth = df[df["response"] == "max_depth_m"].copy()
    if df_depth.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, node in zip(axes, NODES):
        sub = df_depth[df_depth["node"] == node]
        if sub.empty:
            continue

        agg = sub.groupby("param")["interaction"].mean()
        agg = agg.sort_values(ascending=False)

        colors = ["#C44E52" if v > 0.05 else "#8C8C8C" for v in agg.values]
        ax.barh(range(len(agg)), agg.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index, fontsize=9)
        ax.set_xlabel("Interaction Index (ST − S1)")
        ax.set_title(f"{node} — Interaction Effects", fontsize=12)
        ax.axvline(0.05, ls="--", color="gray", lw=0.8, alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_dir / "sobol_interaction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sobol_vs_prcc(df_sobol, df_prcc, out_dir):
    """Scatter: Sobol S1 vs PRCC² for consistency check."""
    if df_prcc is None:
        return

    # Merge on event/node/response/param
    df_s = df_sobol[df_sobol["response"] == "max_depth_m"][
        ["event", "node", "param", "S1"]
    ].copy()
    df_p = df_prcc[df_prcc["response"] == "max_depth_m"][
        ["event", "node", "param", "PRCC"]
    ].copy()
    df_p["PRCC_sq"] = df_p["PRCC"] ** 2

    merged = pd.merge(df_s, df_p, on=["event", "node", "param"], how="inner")
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for node, marker in zip(NODES, ["o", "s"]):
        sub = merged[merged["node"] == node]
        ax.scatter(sub["PRCC_sq"], sub["S1"], label=node,
                   marker=marker, alpha=0.6, s=40)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("PRCC²", fontsize=11)
    ax.set_ylabel("Sobol S1", fontsize=11)
    ax.set_title("Sobol S1 vs PRCC² — max_depth_m", fontsize=12)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(out_dir / "sobol_vs_prcc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # [1/6 | 17%] Load LHS parameters
    print("[1/6 | 17%] Loading LHS parameters...")
    lhs = pd.read_csv(LHS_CSV)
    param_cols = [c for c in lhs.columns if c != "sample"]
    X_all = lhs[param_cols].values
    n_params = len(param_cols)
    print(f"  Parameters ({n_params}): {param_cols}")
    print(f"  LHS samples: {len(lhs)}")

    # Parameter bounds from LHS data
    bounds = [[float(X_all[:, j].min()), float(X_all[:, j].max())]
              for j in range(n_params)]

    problem = {
        "num_vars": n_params,
        "names": param_cols,
        "bounds": bounds,
    }

    # [2/6 | 33%] Generate Saltelli samples
    print(f"[2/6 | 33%] Generating Saltelli samples (N_base={N_SALTELLI})...")
    X_saltelli = saltelli_sampler.sample(problem, N_SALTELLI, calc_second_order=False)
    print(f"  Total Saltelli samples: {X_saltelli.shape[0]:,}")

    # [3/6 | 50%] Fit GP surrogates & compute Sobol
    print("[3/6 | 50%] Fitting GP surrogates & computing Sobol indices...")
    all_results = []
    cv_results  = []

    # Count valid tasks
    tasks = []
    for ev in EVENTS:
        for node in NODES:
            csv_path = SUMMARY_DIR / ev / f"{node}_summary.csv"
            if csv_path.exists():
                tasks.append((ev, node, csv_path))

    pbar = tqdm(total=len(tasks) * len(RESPONSES), desc="Sobol")

    for ev, node, csv_path in tasks:
        df_sum = pd.read_csv(csv_path)
        merged = pd.merge(lhs, df_sum, on="sample", how="inner")
        X = merged[param_cols].values

        for resp in RESPONSES:
            pbar.set_description(f"{ev}-{node}-{resp}")

            if resp not in merged.columns:
                pbar.update(1)
                continue

            y = merged[resp].values
            mask = ~np.isnan(y)
            if mask.sum() < 50 or np.std(y[mask]) < 1e-10:
                pbar.update(1)
                continue

            X_clean, y_clean = X[mask], y[mask]

            try:
                Si, cv_r2 = fit_gp_and_sobol(
                    X_clean, y_clean, problem, X_saltelli
                )
            except Exception as e:
                tqdm.write(f"  [WARN] {ev}-{node}-{resp}: {e}")
                pbar.update(1)
                continue

            cv_results.append({
                "event": ev, "node": node, "response": resp,
                "CV_R2": round(cv_r2, 4),
            })

            for j, pname in enumerate(param_cols):
                all_results.append({
                    "event": ev, "node": node, "response": resp,
                    "param": pname,
                    "S1":      round(Si["S1"][j], 6),
                    "S1_conf": round(Si["S1_conf"][j], 6),
                    "ST":      round(Si["ST"][j], 6),
                    "ST_conf": round(Si["ST_conf"][j], 6),
                    "interaction": round(Si["ST"][j] - Si["S1"][j], 6),
                    "GP_CV_R2": round(cv_r2, 4),
                })

            pbar.update(1)

    pbar.close()

    # [4/6 | 67%] Save results
    print("[4/6 | 67%] Saving results...")
    df_sobol = pd.DataFrame(all_results)
    df_sobol.to_csv(OUT_DIR / "sobol_indices.csv",
                    index=False, encoding="utf-8-sig")
    print(f"  → {OUT_DIR / 'sobol_indices.csv'} ({len(df_sobol)} rows)")

    df_cv = pd.DataFrame(cv_results)
    df_cv.to_csv(OUT_DIR / "gp_cv_scores.csv",
                 index=False, encoding="utf-8-sig")
    print(f"  → {OUT_DIR / 'gp_cv_scores.csv'} ({len(df_cv)} rows)")

    # [5/6 | 83%] Visualizations
    print("[5/6 | 83%] Creating visualizations...")
    plot_s1_st_bar(df_sobol, OUT_DIR)
    print("  → sobol_S1_ST_barplot.png")

    plot_heatmap(df_sobol, OUT_DIR)
    print("  → sobol_heatmap.png")

    plot_interaction(df_sobol, OUT_DIR)
    print("  → sobol_interaction.png")

    # Load PRCC for comparison
    df_prcc = load_prcc_data()
    plot_sobol_vs_prcc(df_sobol, df_prcc, OUT_DIR)
    print("  → sobol_vs_prcc.png")

    # [6/6 | 100%] Summary
    print("\n[6/6 | 100%] Done.")
    print(f"  Output directory: {OUT_DIR}")

    # Print quick summary for max_depth_m
    df_depth = df_sobol[df_sobol["response"] == "max_depth_m"]
    if not df_depth.empty:
        print("\n===== max_depth_m: Mean Sobol Indices (across events) =====")
        for node in NODES:
            sub = df_depth[df_depth["node"] == node]
            if sub.empty:
                continue
            agg = sub.groupby("param")[["S1", "ST", "interaction"]].mean()
            agg = agg.sort_values("ST", ascending=False)
            print(f"\n  [{node}]")
            print(f"  {'Param':<16s} {'S1':>8s} {'ST':>8s} {'ST-S1':>8s}")
            print(f"  {'-'*42}")
            for p, row in agg.iterrows():
                s1v = row['S1']
                stv = row['ST']
                intv = row['interaction']
                print(f"  {p:<16s} {s1v:8.4f} {stv:8.4f} {intv:8.4f}")


if __name__ == "__main__":
    main()
