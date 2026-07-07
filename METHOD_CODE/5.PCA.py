# -*- coding: utf-8 -*-
# 5.PCA.py
# ------------------------------------------------------------
# Unified PCA pipeline (no baseline removal, with scaler floor)
# Step 1: LHS simulated depth -> resample to 480 points (no baseline removal)
# Step 2: Z-score (StandardScaler, scale_floor=0.01) -> PCA fit (95% variance)
# Step 3: PC selection -> Y_{event}_selected.csv + EVR weights
# ------------------------------------------------------------
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ===== USER CONFIG: change paths here =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

# Input: per-event, per-node LHS simulated depth series (sample_*.csv)
LHS_DEPTH_DIR = BASE / "LHS_DEPTH"

# Output: PCA models, PC scores, Y_{event}_selected.csv, PC_selected_evr.csv
OUT_ROOT = BASE / "PCA"

EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

N_POINTS             = 480
VAR_EXPLAINED_TARGET = 0.95
SCALE_FLOOR          = 0.01   # scaler std minimum (sensor precision ~1cm)
N_TOTAL_STEPS        = 3


# ===== STEP 1: Resample =====
def detect_cols(df: pd.DataFrame):
    for tc in ["t", "time", "Time", "timestamp"]:
        if tc in df.columns: break
    else:
        raise RuntimeError(f"Time column not found in {df.columns.tolist()}")
    for dc in ["Depth", "depth", "value", "Depth_m"]:
        if dc in df.columns: break
    else:
        raise RuntimeError(f"Depth column not found in {df.columns.tolist()}")
    return tc, dc


def resample_series(csv_path: Path) -> np.ndarray:
    """Read one sample CSV -> NaN interp -> clip neg -> resample to N_POINTS."""
    df = pd.read_csv(csv_path)
    t_col, d_col = detect_cols(df)

    df[t_col] = pd.to_datetime(df[t_col]).astype("datetime64[ns]")
    df = df.sort_values(t_col).drop_duplicates(subset=t_col, keep="first")

    t = df[t_col].to_numpy()
    y = pd.to_numeric(df[d_col], errors="coerce").to_numpy()

    # NaN interpolation
    if np.all(np.isnan(y)):
        y = np.zeros_like(y, dtype=float)
    else:
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

    n = len(y)
    if n == 0:
        return np.zeros(N_POINTS, dtype=float)

    y = np.maximum(y, 0.0)

    if n == N_POINTS:
        return y

    t0, t1 = t[0], t[-1]
    if t1 == t0:
        s = np.linspace(0.0, 1.0, n)
    else:
        dt_sec = (t1 - t0) / np.timedelta64(1, "s")
        s = ((t - t0) / np.timedelta64(1, "s")) / dt_sec

    return np.interp(np.linspace(0.0, 1.0, N_POINTS), s, y)


def step1_resample() -> dict:
    """Resample all LHS samples. Returns {(event, node): (X_matrix, sample_ids)}"""
    print(f"\n[1/{N_TOTAL_STEPS} | 33%] Resampling LHS depth to {N_POINTS} points...")
    data = {}

    for event in EVENTS:
        for node in NODES:
            in_dir = LHS_DEPTH_DIR / event / node
            if not in_dir.exists():
                print(f"  [WARN] Missing: {in_dir}")
                continue

            csvs = sorted(in_dir.glob("sample_*.csv"))
            rows, ids = [], []

            for csv_path in tqdm(csvs, desc=f"  {event}-{node}", leave=False):
                try:
                    y = resample_series(csv_path)
                    rows.append(y)
                    ids.append(csv_path.stem)
                except Exception as e:
                    print(f"  [ERR] {csv_path.name}: {e}")

            if rows:
                data[(event, node)] = (np.vstack(rows), ids)
                tqdm.write(f"  [OK] {event}-{node}: {len(rows)} samples")

    return data


# ===== STEP 2: Z-score (with floor) + PCA fit =====
def step2_pca_fit(data: dict) -> dict:
    """Fit StandardScaler (floored) + PCA per node."""
    print(f"\n[2/{N_TOTAL_STEPS} | 66%] Z-score (floor={SCALE_FLOOR}) + PCA fitting...")
    models = {}

    for node in NODES:
        X_list = []
        meta = []
        for event in EVENTS:
            key = (event, node)
            if key not in data:
                continue
            X_ev, ids = data[key]
            X_list.append(X_ev)
            meta.extend([(event, sid) for sid in ids])

        if not X_list:
            print(f"  [WARN] No data for {node}")
            continue

        X_all = np.vstack(X_list)

        # Z-score fit
        scaler = StandardScaler()
        scaler.fit(X_all)

        # Apply floor to scale_ (prevent division by near-zero std)
        n_floored = int((scaler.scale_ < SCALE_FLOOR).sum())
        scaler.scale_ = np.maximum(scaler.scale_, SCALE_FLOOR)
        print(f"  [{node}] scale_floor applied: {n_floored} features clamped (of {len(scaler.scale_)})")

        # Transform with floored scaler
        X_scaled = scaler.transform(X_all)

        # PCA fit (95% cumulative variance)
        pca = PCA(n_components=VAR_EXPLAINED_TARGET, svd_solver="full")
        X_pca = pca.fit_transform(X_scaled)

        n_pc = pca.n_components_
        evr = pca.explained_variance_ratio_
        cum_evr = np.cumsum(evr)

        print(f"  [{node}] n_samples={X_all.shape[0]}, n_PC={n_pc}, "
              f"cum_EVR={cum_evr[-1]:.4f}")
        for i in range(n_pc):
            print(f"    PC{i+1}: EVR={evr[i]:.4f} (cum={cum_evr[i]:.4f})")

        # Save models
        model_dir = OUT_ROOT / "models" / node
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(model_dir / "pca.pkl", "wb") as f:
            pickle.dump(pca, f)

        pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(n_pc)],
            "explained_variance_ratio": evr,
            "cumulative_evr": cum_evr,
        }).to_csv(model_dir / "explained_variance.csv", index=False)

        # Save per-event PC scores
        scores_dir = OUT_ROOT / "scores"
        by_event = defaultdict(list)
        for (ev, sid), row in zip(meta, X_pca):
            rec = {"sample": sid}
            for i in range(n_pc):
                rec[f"PC{i+1}"] = float(row[i])
            by_event[ev].append(rec)

        for ev, rows in by_event.items():
            ev_dir = scores_dir / ev / node
            ev_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                ev_dir / f"PCA_scores_{ev}_{node}.csv", index=False)

        evr_dict = {f"{node}_PC{i+1}": float(evr[i]) for i in range(n_pc)}
        models[node] = (scaler, pca, evr_dict)

    return models


# ===== STEP 3: PC selection + Y_event + EVR weights =====
def step3_select(models: dict):
    print(f"\n[3/{N_TOTAL_STEPS} | 100%] PC selection and Y matrix assembly...")

    scores_dir = OUT_ROOT / "scores"
    evr_records = []
    node_pc_map = {}

    for node in NODES:
        if node not in models:
            continue
        _, pca, evr_dict = models[node]
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)

        k = int(np.searchsorted(cum, VAR_EXPLAINED_TARGET) + 1)
        k = max(1, min(k, len(evr)))

        targets = [f"{node}_PC{i+1}" for i in range(k)]
        node_pc_map[node] = targets

        for t in targets:
            evr_records.append({"target": t, "evr": evr_dict[t]})

        print(f"  [{node}] Selected: {[f'PC{i+1}' for i in range(k)]}, "
              f"cum_EVR={cum[k-1]:.4f}")

    for ev in EVENTS:
        merged = None
        for node, targets in node_pc_map.items():
            score_path = scores_dir / ev / node / f"PCA_scores_{ev}_{node}.csv"
            if not score_path.exists():
                print(f"  [WARN] Missing: {score_path}")
                continue

            df_s = pd.read_csv(score_path)
            want_pcs = [t.split("_", 1)[1] for t in targets]
            keep = ["sample"] + [pc for pc in df_s.columns if pc in want_pcs]
            df_s = df_s[keep].rename(
                columns={pc: f"{node}_{pc}" for pc in want_pcs if pc in df_s.columns})

            merged = df_s if merged is None else merged.merge(df_s, on="sample", how="inner")

        if merged is not None:
            merged = merged.sort_values("sample").reset_index(drop=True)
            out_path = OUT_ROOT / f"Y_{ev}_selected.csv"
            merged.to_csv(out_path, index=False)
            print(f"  [OK] Y_{ev}_selected.csv: {merged.shape}")

    df_evr = pd.DataFrame(evr_records)
    evr_path = OUT_ROOT / "PC_selected_evr.csv"
    df_evr.to_csv(evr_path, index=False)
    print(f"  [OK] PC_selected_evr.csv: {len(evr_records)} PCs")


# ===== MAIN =====
def main():
    print("=" * 70)
    print(" PCA Pipeline (no baseline removal, scale_floor=0.01)")
    print(f" Input:  {LHS_DEPTH_DIR}")
    print(f" Output: {OUT_ROOT}")
    print("=" * 70)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    data = step1_resample()
    models = step2_pca_fit(data)
    step3_select(models)

    print("\n" + "=" * 70)
    print(" PCA Pipeline DONE!")
    print(f" Output: {OUT_ROOT}")
    print("=" * 70)
    print("\nOutput structure:")
    print(f"  {OUT_ROOT}/")
    print(f"    models/{{M0113,MH0126}}/  <- scaler.pkl, pca.pkl, explained_variance.csv")
    print(f"    scores/{{R1..R6}}/{{node}}/ <- PCA_scores_*.csv")
    print(f"    Y_R*_selected.csv          <- merged PC scores per event")
    print(f"    PC_selected_evr.csv        <- EVR weights for objective function")


if __name__ == "__main__":
    main()
