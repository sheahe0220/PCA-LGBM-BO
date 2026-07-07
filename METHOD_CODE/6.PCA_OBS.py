# -*- coding: utf-8 -*-
# 6.PCA_OBS.py
# ------------------------------------------------------------
# Min-baseline OBS -> 480pt resample -> scaler/pca transform
# -> obs_pc_{event}.csv
# Uses the PCA models fitted on simulations in 5.PCA.py (unchanged)
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== USER CONFIG: change paths here =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

# Input: min-baseline corrected observations (Depth_adjusted per event-node)
OBS_DIR = BASE / "OBS_min_baseline"

# Input: PCA models fitted on simulations in 5.PCA.py
PCA_ROOT  = BASE / "PCA"
MODEL_DIR = PCA_ROOT / "models"

# Output: observed PC scores
OUT_DIR = PCA_ROOT / "obs_pc"

EVENTS   = ["R1", "R2", "R5", "R6", "R8"]
NODES    = ["M0113", "MH0126"]
N_POINTS = 480


def resample_to_n(x: np.ndarray, n_target: int) -> np.ndarray:
    n = len(x)
    if n == n_target:
        return x
    if n == 0:
        return np.zeros(n_target, dtype=float)
    orig = np.linspace(0.0, 1.0, n)
    target = np.linspace(0.0, 1.0, n_target)
    return np.interp(target, orig, x)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    N_TOTAL = 3

    # [1/3 | 33%] Load PCA models
    print(f"[1/{N_TOTAL} | 33%] Loading PCA models...")
    models = {}
    for node in NODES:
        scaler_path = MODEL_DIR / node / "scaler.pkl"
        pca_path = MODEL_DIR / node / "pca.pkl"
        if not scaler_path.exists() or not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {node}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        models[node] = (scaler, pca)
        print(f"  [{node}] scaler + pca loaded (n_PC={pca.n_components_})")

    # [2/3 | 66%] Load EVR
    print(f"\n[2/{N_TOTAL} | 66%] Loading EVR weights...")
    evr_path = PCA_ROOT / "PC_selected_evr.csv"
    df_evr = pd.read_csv(evr_path)
    selected_targets = df_evr["target"].tolist()
    print(f"  Selected PCs: {selected_targets}")

    # [3/3 | 100%] Transform min-baseline OBS per event
    print(f"\n[3/{N_TOTAL} | 100%] Transforming min-baseline OBS...")
    summary = []

    for event in tqdm(EVENTS, desc="Events"):
        obs_pcs = {}

        for node in NODES:
            obs_path = OBS_DIR / event / f"{node}_obs_depth_{event}_minbase.csv"
            if not obs_path.exists():
                print(f"  [WARN] Missing: {obs_path}")
                continue

            df = pd.read_csv(obs_path)
            depth = df["Depth_adjusted"].to_numpy(dtype=float)

            if np.any(np.isnan(depth)):
                depth = pd.Series(depth).interpolate(limit_direction="both").to_numpy()

            x_feat = resample_to_n(depth, N_POINTS)

            scaler, pca = models[node]
            x_scaled = scaler.transform(x_feat.reshape(1, -1))
            x_pc = pca.transform(x_scaled)[0]

            n_pc = len(x_pc)
            for i in range(n_pc):
                pc_name = f"{node}_PC{i+1}"
                obs_pcs[pc_name] = float(x_pc[i])

        if obs_pcs:
            df_out = pd.DataFrame([obs_pcs])
            out_path = OUT_DIR / f"obs_pc_{event}.csv"
            df_out.to_csv(out_path, index=False)

            for k, v in obs_pcs.items():
                summary.append({"event": event, "target": k, "pc_score": v})

            tqdm.write(f"  [OK] {event}: {list(obs_pcs.keys())} -> {out_path.name}")

    if summary:
        df_sum = pd.DataFrame(summary)
        df_sum.to_csv(OUT_DIR / "obs_pc_summary.csv", index=False)

    print(f"\n[DONE] Min-baseline obs PC scores saved to: {OUT_DIR}")
    print(f"  Files: {[f'obs_pc_{ev}.csv' for ev in EVENTS]}")


if __name__ == "__main__":
    main()
