
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# ===============================
# Base paths
# ===============================
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

# Observations aligned to the model timeline
ALIGNED_DIR = BASE / "input" / "obs_depth" / "aligned_to_LHS"

# LHS-based PCA models
PCA_MODEL_DIR = BASE / "PCA" / "PCA_models_1"

# Output directory for observed PCA scores
OBS_PCA_DIR = BASE / "PCA" / "obs_pca"
OBS_PCA_DIR.mkdir(parents=True, exist_ok=True)

# Events / nodes
EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

# Baseline settings (consistent with the LHS pipeline concept)
BASELINE_FRAC = 0.10   # median over the first 10% segment
CLIP_NEGATIVE = True   # after baseline removal, clip negatives to 0


# ===============================
# Utilities
# ===============================
def load_pca_and_scaler(node_id: str):
    node_dir = PCA_MODEL_DIR / node_id
    pca_path = node_dir / "pca.pkl"
    scl_path = node_dir / "scaler.pkl"

    if not pca_path.exists():
        raise FileNotFoundError(f"[ERR] PCA file not found: {pca_path}")
    if not scl_path.exists():
        raise FileNotFoundError(f"[ERR] Scaler file not found: {scl_path}")

    with open(pca_path, "rb") as f:
        pca = pickle.load(f)
    with open(scl_path, "rb") as f:
        scaler = pickle.load(f)

    return pca, scaler


def preprocess_obs_series(x_raw: np.ndarray, n_target: int) -> np.ndarray:
    # 1) NaN handling
    if np.all(np.isnan(x_raw)):
        x = np.zeros_like(x_raw, dtype=float)
    else:
        s = pd.Series(x_raw.astype(float))
        s = s.interpolate(limit_direction="both")
        x = s.to_numpy()

    L = x.shape[0]
    if L == 0:
        return np.zeros(n_target, dtype=float)

    # 2) Baseline (median over the first 10% segment)
    head_n = max(1, int(L * BASELINE_FRAC))
    base_region = x[:head_n]
    if np.any(~np.isnan(base_region)):
        baseline = float(np.nanmedian(base_region))
    else:
        baseline = float(np.nanmedian(x))

    x = x - baseline

    # 3) Clip negatives
    if CLIP_NEGATIVE:
        x = np.maximum(x, 0.0)

    # 4) Resample to n_target (index-based 0~1 -> 0~1)
    if L == n_target:
        return x

    orig_pos = np.linspace(0.0, 1.0, L)
    new_pos  = np.linspace(0.0, 1.0, n_target)
    x_resampled = np.interp(new_pos, orig_pos, x)

    return x_resampled


def compute_obs_pca_for_node(node_id: str):
    print(f"\n[INFO] Start processing node: {node_id}")

    # Load PCA/scaler
    pca, scaler = load_pca_and_scaler(node_id)
    n_pc      = pca.n_components_
    n_feature = pca.n_features_in_  # typically 480
    pc_cols   = [f"PC{i+1}" for i in range(n_pc)]

    records = []

    for event in EVENTS:
        csv_path = ALIGNED_DIR / f"{event}_{node_id}_obs_depth_aligned_to_sample001.csv"
        if not csv_path.exists():
            print(f"  [WARN] File not found, skipped: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if "Depth_obs" not in df.columns:
            raise RuntimeError(f"[ERR] Missing 'Depth_obs' column in: {csv_path}")

        # Raw observed series
        x_raw = df["Depth_obs"].values

        # Preprocess + resample to match PCA input feature length
        x_feat = preprocess_obs_series(x_raw, n_feature)

        # Scale + PCA projection
        x_feat_2d = x_feat.reshape(1, -1)
        x_scaled  = scaler.transform(x_feat_2d)
        x_pc      = pca.transform(x_scaled)

        rec = {"event": event}
        for i, col in enumerate(pc_cols):
            rec[col] = float(x_pc[0, i])
        records.append(rec)

        print(f"  [OK] {event}-{node_id}: Observed PCA computed.")

    if not records:
        print(f"[WARN] {node_id}: No observed PCA results.")
        return

    df_out = pd.DataFrame(records)
    out_path = OBS_PCA_DIR / f"{node_id}_obs_pca_scores.csv"
    df_out.to_csv(out_path, index=False)
    print(f"[SAVE] {node_id}: {out_path}")


def main():
    for node in NODES:
        compute_obs_pca_for_node(node)

    print("\n[DONE] Observed PCA computation finished for all nodes.")


if __name__ == "__main__":
    main()
