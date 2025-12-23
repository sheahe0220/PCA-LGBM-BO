from pathlib import Path
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== Base paths =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

# Input paths
PARAMS_CSV = BASE / "input" / "lhs_samples_4d.csv"
PCA_DIR = BASE / "PCA"
OBS_PC_DIR = PCA_DIR / "obs_pc_1"

# Output paths
OUT_ROOT = BASE / "COMPARE" / "1_"
MODEL_DIR = OUT_ROOT / "models" / "pooled"
RESULT_DIR = OUT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Settings =====
EVENTS = ["R1", "R2", "R5", "R6"]
FEATURES = ["Imperv", "Width", "Nimp", "n_pipe"]  # Parameters to predict

# LightGBM base settings
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

N_FOLDS = 5
EARLY_STOP = 100


# ===== Utility functions =====
def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]


def sample_id_normalize(s: str) -> str:
    import re
    m = re.search(r"(\d+)$", str(s))
    return f"sample_{int(m.group(1)):03d}" if m else str(s)


# ===== Data loaders =====
def load_lhs_params() -> pd.DataFrame:
    if not PARAMS_CSV.exists():
        raise FileNotFoundError(f"LHS parameter file not found: {PARAMS_CSV}")

    df = drop_unnamed(pd.read_csv(PARAMS_CSV))

    # Unify the ID column name
    if "run_id" in df.columns:
        id_col = "run_id"
    elif "sample" in df.columns:
        id_col = "sample"
    else:
        raise RuntimeError("PARAMS_CSV must contain either 'run_id' or 'sample' column")

    df = df.rename(columns={id_col: "sample"})
    df["sample"] = df["sample"].apply(sample_id_normalize)

    # Check required parameter columns
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise RuntimeError(f"Missing parameter columns: {missing}")

    return df[["sample"] + FEATURES]


def load_all_model_pc() -> pd.DataFrame:
    all_data = []

    for event in EVENTS:
        path = PCA_DIR / f"Y_{event}_selected.csv"
        if not path.exists():
            raise FileNotFoundError(f"Model PC file not found: {path}")

        df = drop_unnamed(pd.read_csv(path))

        # Unify sample column
        if "run_id" in df.columns:
            df = df.rename(columns={"run_id": "sample"})
        if "sample" not in df.columns:
            raise RuntimeError(f"Missing 'sample' column in: {path}")

        df["sample"] = df["sample"].apply(sample_id_normalize)
        df["event"] = event

        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_all_obs_pc() -> pd.DataFrame:
    all_obs = []

    for event in EVENTS:
        path = OBS_PC_DIR / f"obs_pc_{event}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Observed PC file not found: {path}")

        df = drop_unnamed(pd.read_csv(path))

        # If multiple rows exist, use the first one only
        if len(df) > 1:
            df = df.iloc[[0]]

        df["event"] = event
        all_obs.append(df)

    combined = pd.concat(all_obs, ignore_index=True)
    return combined


# ===== Training stage: pooled PC → θ =====
def train_pooled():
    print("=" * 80)
    print("Case A-Pooled: Training LightGBM for pooled PC → θ")
    print("=" * 80)

    df_params = load_lhs_params()
    df_pc_all = load_all_model_pc()

    # Merge by sample
    merged = pd.merge(df_pc_all, df_params, on="sample", how="inner")

    if merged.empty:
        raise RuntimeError("Merge failed: merged dataframe is empty (PC vs parameters)")

    # X: PC columns only
    pc_cols = [c for c in merged.columns if "_PC" in c.upper()]
    if not pc_cols:
        raise RuntimeError("No PC columns found (must include '_PC' in column name)")

    X = merged[pc_cols].astype(float)

    print(f"Pooled training data: N={len(X)}, PC_dim={len(pc_cols)}")
    print(f"PC columns used: {pc_cols}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train one model per parameter
    results = []

    for param in FEATURES:
        print(f"\n[TRAIN] Training model for parameter: '{param}'")
        y = merged[param].astype(float)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        best_iters = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
            )

            best_iter = model.best_iteration_ or LGBM_PARAMS["n_estimators"]
            best_iters.append(best_iter)

            # OOF prediction
            oof_preds[val_idx] = model.predict(X_val, num_iteration=best_iter)

            # Save model
            model_path = MODEL_DIR / f"{param}_fold{fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=best_iter)

        # CV metrics
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        mae = mean_absolute_error(y, oof_preds)
        r2 = r2_score(y, oof_preds)

        print(f"  CV metrics: RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.4f}")

        results.append({
            "param": param,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_iter_mean": float(np.mean(best_iters)),
        })

    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(RESULT_DIR / "pooled_cv_metrics.csv", index=False)

    print("\n[DONE] Training finished: pooled PC → θ")
    print(f"  - CV summary saved: {RESULT_DIR / 'pooled_cv_metrics.csv'}")


# ===== Prediction stage: event-wise θ̂_e + θ-space pooling =====
def predict_eventwise_and_pooled():
    print("\n" + "=" * 80)
    print("Case A-Pooled: Predicting θ̂_e per event and pooling in θ-space")
    print("=" * 80)

    # Load all observed PCs
    df_obs_all = load_all_obs_pc()
    pc_cols = [c for c in df_obs_all.columns if "_PC" in c.upper()]

    if not pc_cols:
        raise RuntimeError("No '_PC' columns found in observed PC data")

    print(f"Observed PC columns: {pc_cols}")

    eventwise_theta = {}  # {event: {param: value}}

    for event in EVENTS:
        df_ev = df_obs_all[df_obs_all["event"] == event]
        if df_ev.empty:
            print(f"[WARN] No observed PC row for {event}, skipped")
            continue

        row = df_ev.iloc[0]
        X_obs = row[pc_cols].astype(float).values.reshape(1, -1)

        print(f"\n[{event}] Predicting θ̂_e from observed PCs")

        theta_hat_e = {}

        for param in FEATURES:
            preds = []
            for fold in range(1, N_FOLDS + 1):
                model_path = MODEL_DIR / f"{param}_fold{fold}.txt"
                if not model_path.exists():
                    print(f"  [WARN] Model file not found: {model_path}")
                    continue

                bst = lgb.Booster(model_file=str(model_path))
                pred = bst.predict(X_obs)[0]
                preds.append(pred)

            if preds:
                mean_pred = float(np.mean(preds))
                std_pred = float(np.std(preds))
                theta_hat_e[param] = mean_pred
                print(f"  {param}: {mean_pred:.6f} (±{std_pred:.6f})")
            else:
                theta_hat_e[param] = np.nan
                print(f"  {param}: prediction failed (no model files)")

        eventwise_theta[event] = theta_hat_e

    # Save event-wise θ̂_e
    if eventwise_theta:
        with open(RESULT_DIR / "pooled_eventwise_theta_predictions.json", "w", encoding="utf-8") as f:
            json.dump(eventwise_theta, f, indent=2, ensure_ascii=False)

        df_ev_pred = pd.DataFrame(eventwise_theta).T
        df_ev_pred.index.name = "event"
        df_ev_pred.to_csv(RESULT_DIR / "pooled_eventwise_theta_predictions.csv")

        print("\n[INFO] Event-wise θ̂_e predictions:")
        print(df_ev_pred)
    else:
        print("[ERROR] No θ̂_e obtained for any event. θ-space pooling cannot proceed.")
        return

    # θ-space pooling: mean over events
    theta_pooled = {}

    print("\n" + "-" * 80)
    print("θ-space pooling: mean of event-wise θ̂_e → θ̂_pooled")
    print("-" * 80)

    for param in FEATURES:
        vals = []
        for event in EVENTS:
            if event in eventwise_theta and param in eventwise_theta[event]:
                val = eventwise_theta[event][param]
                if np.isfinite(val):
                    vals.append(val)

        if vals:
            pooled_val = float(np.mean(vals))
            theta_pooled[param] = pooled_val
            print(f"  {param}: mean({[round(v, 6) for v in vals]}) = {pooled_val:.6f}")
        else:
            theta_pooled[param] = np.nan
            print(f"  {param}: set to NaN (no valid event-wise predictions)")

    # Save θ̂_pooled
    df_theta = pd.DataFrame([theta_pooled])
    df_theta.index = ["theta_pooled"]
    df_theta.to_csv(RESULT_DIR / "pooled_theta_optimal.csv")

    with open(RESULT_DIR / "pooled_theta_optimal.json", "w", encoding="utf-8") as f:
        json.dump(theta_pooled, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Pooled optimal parameter set θ̂_pooled (mean in θ-space):")
    print(df_theta)
    print("=" * 80)

    print(f"\nOutputs saved to: {RESULT_DIR}")
    print("  - pooled_eventwise_theta_predictions.csv / .json")
    print("  - pooled_theta_optimal.csv / .json")
    print("  - pooled_cv_metrics.csv")


# ===== Main =====
def main():
    train_pooled()
    predict_eventwise_and_pooled()


if __name__ == "__main__":
    main()
