from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===============================
# Paths / Settings
# ===============================
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

# Resampled outputs generated in Step 1
RESAMPLED_BASE = BASE / "PCA" / "PCA_resampled_1"

# Where to save PCA models / scores
PCA_MODEL_BASE  = BASE / "PCA" / "PCA_models_1"
PCA_SCORES_BASE = BASE / "PCA" / "PCA_scores_1"

EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

PCA_MODEL_BASE.mkdir(parents=True, exist_ok=True)
PCA_SCORES_BASE.mkdir(parents=True, exist_ok=True)

# Cumulative explained variance target (e.g., 95%)
VAR_EXPLAINED_TARGET = 0.95


def load_resampled_for_node(node_id: str):
    all_rows = []
    meta = []  # (event, sample)

    for ev in EVENTS:
        csv_path = RESAMPLED_BASE / ev / node_id / f"PCA_input_{ev}_{node_id}.csv"
        if not csv_path.exists():
            print(f"[WARN] Resampled file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "sample" not in df.columns:
            raise RuntimeError(f"[ERR] Missing 'sample' column in: {csv_path}")

        feature_cols = [c for c in df.columns if c != "sample"]
        X = df[feature_cols].to_numpy(dtype=float)

        for i, sample_id in enumerate(df["sample"].tolist()):
            all_rows.append(X[i, :])
            meta.append((ev, sample_id))

    if not all_rows:
        raise RuntimeError(f"[ERR] No data loaded for node: {node_id}")

    X_all = np.vstack(all_rows)
    return X_all, meta


def fit_pca_for_node(node_id: str):
    # 1) Aggregate LHS + resampled data across all events
    X_all, meta = load_resampled_for_node(node_id)

    # 2) Z-score standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # 3) Fit PCA until cumulative explained variance reaches the target
    pca = PCA(n_components=VAR_EXPLAINED_TARGET, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    n_pc = pca.n_components_
    print(
        f"[{node_id}] PCA fitted: n_samples={X_all.shape[0]}, "
        f"n_features={X_all.shape[1]}, n_PC={n_pc}"
    )

    # 4) Save models (scaler + pca)
    model_dir = PCA_MODEL_BASE / node_id
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(model_dir / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    # 5) Save explained variance ratios
    evr = pca.explained_variance_ratio_
    df_evr = pd.DataFrame(
        {
            "PC": [f"PC{i+1}" for i in range(n_pc)],
            "explained_variance_ratio": evr,
        }
    )
    df_evr.to_csv(model_dir / "explained_variance.csv", index=False)

    # 6) Write event-wise PCA score files
    from collections import defaultdict
    by_event = defaultdict(list)

    for (ev, sample_id), row in zip(meta, X_pca):
        rec = {"sample": sample_id}
        for i in range(n_pc):
            rec[f"PC{i+1}"] = float(row[i])
        by_event[ev].append(rec)

    for ev, rows in by_event.items():
        out_dir = PCA_SCORES_BASE / ev / node_id
        out_dir.mkdir(parents=True, exist_ok=True)
        df_ev = pd.DataFrame(rows)
        df_ev.to_csv(out_dir / f"PCA_scores_{ev}_{node_id}.csv", index=False)
        print(f"[OK] {node_id}-{ev}: PCA scores saved -> {out_dir}")


def main():
    for node in NODES:
        fit_pca_for_node(node)

    print("\n[DONE] Z-score + PCA stage finished.")


if __name__ == "__main__":
    main()
