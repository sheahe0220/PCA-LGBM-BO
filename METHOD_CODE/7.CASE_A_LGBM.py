# -*- coding: utf-8 -*-
# 7.CASE_A_LGBM.py
# ------------------------------------------------------------
# Case A - Inverse LightGBM: PC scores (8D) -> theta (4 params)
# Uses min-baseline OBS PCA
# Input:  PCA/Y_R*_selected.csv, PCA/obs_pc/obs_pc_summary.csv,
#         input/lhs_samples_4d.csv
# Output: RESULTS/CaseA_LGBM/models/, RESULTS/CaseA_LGBM/results/
# ------------------------------------------------------------
from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# ===== USER CONFIG: change paths here =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

PARAMS_CSV = BASE / "input" / "lhs_samples_4d.csv"
PCA_DIR    = BASE / "PCA"
OBS_PC_CSV = PCA_DIR / "obs_pc" / "obs_pc_summary.csv"

OUT_ROOT   = BASE / "RESULTS" / "CaseA_LGBM"
MODEL_DIR  = OUT_ROOT / "models"
RESULT_DIR = OUT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Config =====
EVENTS   = ["R1", "R2", "R5", "R6"]
FEATURES = ["Imperv", "Width", "Nimp", "n_pipe"]

LGBM_PARAMS = {
    "boosting_type":    "gbdt",
    "objective":        "regression",
    "metric":           "rmse",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        -1,
    "n_estimators":     2000,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           -1,
}

N_FOLDS    = 5
EARLY_STOP = 100


# ===== Utils =====
def normalize_sample(s):
    m = re.search(r"(\d+)$", str(s))
    return f"sample_{int(m.group(1)):03d}" if m else str(s)


# ===== Data loaders =====
def load_lhs_params() -> pd.DataFrame:
    df = pd.read_csv(PARAMS_CSV)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
    id_col = "run_id" if "run_id" in df.columns else "sample"
    df = df.rename(columns={id_col: "sample"})
    df["sample"] = df["sample"].apply(normalize_sample)
    return df[["sample"] + FEATURES]


def load_all_model_pc() -> pd.DataFrame:
    frames = []
    for ev in tqdm(EVENTS, desc="  Y loading"):
        path = PCA_DIR / f"Y_{ev}_selected.csv"
        if not path.exists():
            raise FileNotFoundError(f"Y file missing: {path}")
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "sample"})
        df["sample"] = df["sample"].apply(normalize_sample)
        df["event"]  = ev
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_obs_pc_summary() -> pd.DataFrame:
    df = pd.read_csv(OBS_PC_CSV)
    wide = df.pivot(index="event", columns="target", values="pc_score").reset_index()
    wide.columns.name = None
    return wide


# ===== STEP 1: Train pooled inverse models =====
def train_pooled():
    print("\n[1/3 | 33%] Pooled PC -> theta LightGBM training")

    df_params = load_lhs_params()
    df_pc_all = load_all_model_pc()

    merged = pd.merge(df_pc_all, df_params, on="sample", how="inner")
    if merged.empty:
        raise RuntimeError("PC-param merge is empty")

    pc_cols = sorted([c for c in merged.columns if "_PC" in c.upper()])
    if not pc_cols:
        raise RuntimeError("No PC columns found")

    X = merged[pc_cols].astype(float)
    print(f"  Training data: N={len(X)}, PC dims={len(pc_cols)}")
    print(f"  PC columns: {pc_cols}")

    results = []
    for param in tqdm(FEATURES, desc="  Training"):
        y = merged[param].astype(float)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof = np.zeros(len(X))
        best_iters = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
            )
            best_iter = model.best_iteration_ or LGBM_PARAMS["n_estimators"]
            best_iters.append(best_iter)
            oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=best_iter)
            model.booster_.save_model(
                str(MODEL_DIR / f"{param}_fold{fold}.txt"),
                num_iteration=best_iter,
            )

        rmse = float(np.sqrt(mean_squared_error(y, oof)))
        mae  = float(mean_absolute_error(y, oof))
        r2   = float(r2_score(y, oof))
        print(f"    [{param}] RMSE={rmse:.6f}  MAE={mae:.6f}  R2={r2:.4f}")
        results.append({
            "param": param, "rmse": rmse, "mae": mae, "r2": r2,
            "best_iter_mean": float(np.mean(best_iters)),
        })

    pd.DataFrame(results).to_csv(RESULT_DIR / "cv_metrics.csv", index=False)
    print("  [OK] cv_metrics.csv saved")


# ===== STEP 2: Event-wise prediction =====
def predict_eventwise() -> dict:
    print("\n[2/3 | 66%] Event-wise theta prediction")

    df_obs = load_obs_pc_summary()
    pc_cols = sorted([c for c in df_obs.columns if "_PC" in c.upper()])
    print(f"  OBS PC columns: {pc_cols}")

    eventwise = {}
    for ev in tqdm(EVENTS, desc="  Events"):
        row = df_obs[df_obs["event"] == ev]
        if row.empty:
            print(f"  [WARN] {ev}: no OBS PC data"); continue

        X_obs = row[pc_cols].astype(float).values.reshape(1, -1)
        theta_e = {}

        for param in FEATURES:
            preds = []
            for fold in range(1, N_FOLDS + 1):
                mp = MODEL_DIR / f"{param}_fold{fold}.txt"
                if not mp.exists():
                    print(f"  [WARN] model missing: {mp}"); continue
                bst = lgb.Booster(model_file=str(mp))
                preds.append(float(bst.predict(X_obs)[0]))

            if preds:
                theta_e[param] = float(np.mean(preds))
                print(f"    [{ev}] {param}: {theta_e[param]:.6f} "
                      f"(+/-{float(np.std(preds)):.6f})")
            else:
                theta_e[param] = float("nan")

        eventwise[ev] = theta_e

    with open(RESULT_DIR / "eventwise_theta.json", "w", encoding="utf-8") as f:
        json.dump(eventwise, f, indent=2, ensure_ascii=False)
    df_ev = pd.DataFrame(eventwise).T
    df_ev.index.name = "event"
    df_ev.to_csv(RESULT_DIR / "eventwise_theta.csv")
    print("  [OK] eventwise_theta.csv saved")
    return eventwise


# ===== STEP 3: Theta pooling =====
def pool_theta(eventwise: dict):
    print("\n[3/3 | 100%] Theta pooling")

    theta_pooled = {}
    for param in FEATURES:
        vals = [eventwise[ev][param] for ev in EVENTS
                if ev in eventwise and np.isfinite(eventwise[ev].get(param, np.nan))]
        theta_pooled[param] = float(np.mean(vals)) if vals else float("nan")
        print(f"  {param}: {[round(v,6) for v in vals]} -> pooled={theta_pooled[param]:.6f}")

    pd.DataFrame([theta_pooled], index=["theta_pooled"]).to_csv(
        RESULT_DIR / "pooled_theta_optimal.csv")
    with open(RESULT_DIR / "pooled_theta_optimal.json", "w", encoding="utf-8") as f:
        json.dump(theta_pooled, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print("Final theta_pooled:")
    for k, v in theta_pooled.items():
        print(f"  {k}: {v:.8f}")
    print(f"{'='*55}")
    print(f"\n  Results: {RESULT_DIR}")


# ===== Main =====
def main():
    print("=" * 55)
    print("LGBM Inverse Training (min-baseline OBS)")
    print(f"  PCA dir:  {PCA_DIR}")
    print(f"  OBS PC:   {OBS_PC_CSV}")
    print(f"  LHS:      {PARAMS_CSV}")
    print(f"  Output:   {OUT_ROOT}")
    print("=" * 55)

    train_pooled()
    eventwise = predict_eventwise()
    pool_theta(eventwise)


if __name__ == "__main__":
    main()
