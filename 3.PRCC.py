# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as t_dist

# ===== Paths / basic settings =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

LHS_CSV     = BASE / "LHS_R" / "lhs_samples_10d.csv"
SUMMARY_DIR = BASE / "LHS_OUT_RESULTS"
PRCC_DIR    = BASE / "PRCC"

# Event / node lists
EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["MH0126", "M0113", "SUM", "M0131", "OUT1", "M9999"]

# Candidate response metrics (used only if present)
RESP_CAND = [
    "max_depth_m",
    "dur_depth_gt_0_5m_min",
    "max_flood_rate_m3s",
    "flood_vol_m3",
    "tot_inflow_vol_m3",
    "avg_depth_m",
    "t_idx_max_depth",
    "max_total_inflow_m3s",
]


# ===== PRCC computation function =====
def compute_prcc_matrix(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    X: (N x p) parameter matrix (DataFrame, columns = parameters)
    y: (N,) response vector (Series)

    Returns:
        DataFrame consisting of
        param, PRCC, t_stat, p_value, N, df
    """
    # Remove rows with missing values (drop if NaN in either X or y)
    data = X.copy()
    data["__y__"] = y.values
    data = data.dropna(axis=0)
    if len(data) < 5:
        raise RuntimeError(f"Too few valid samples (N={len(data)})")

    # Rank transform (PRCC is rank-based partial correlation)
    Z = data.values
    Z_rank = np.zeros_like(Z, dtype=float)
    for j in range(Z.shape[1]):
        Z_rank[:, j] = rankdata(Z[:, j], method="average")

    # Correlation matrix (Pearson) computed on ranked values
    R = np.corrcoef(Z_rank, rowvar=False)
    # Inverse matrix
    C = np.linalg.inv(R)

    p = X.shape[1]            # number of parameters
    n = Z.shape[0]            # number of samples
    df = n - p - 1            # degrees of freedom (two variables + (p−1) controls)

    res_rows = []
    for j, pname in enumerate(X.columns):
        # Partial correlation between X_j and y, controlling all other X
        prcc = -C[j, -1] / np.sqrt(C[j, j] * C[-1, -1])

        # t-statistic and p-value
        if abs(prcc) >= 1.0:
            t_stat = np.sign(prcc) * np.inf
            p_val = 0.0
        else:
            t_stat = prcc * np.sqrt(df / (1.0 - prcc ** 2))
            p_val = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df))

        res_rows.append({
            "param": pname,
            "PRCC": prcc,
            "t_stat": t_stat,
            "p_value": p_val,
            "N": n,
            "df": df,
        })

    return pd.DataFrame(res_rows)


# ===== Main routine =====
def main():
    PRCC_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Read LHS parameters
    lhs = pd.read_csv(LHS_CSV)
    if "sample" not in lhs.columns:
        raise RuntimeError("'sample' column is required in the LHS CSV.")
    param_cols = [c for c in lhs.columns if c != "sample"]

    print(f"[INFO] LHS parameter columns ({len(param_cols)}): {param_cols}")

    # Loop over events
    for ev in EVENTS:
        ev_dir = SUMMARY_DIR / ev
        if not ev_dir.exists():
            print(f"[WARN] Event {ev}: summary directory not found -> {ev_dir}")
            continue

        print(f"\n[EVENT] Start processing {ev}")

        for node in NODES:
            summary_csv = ev_dir / f"{node}_summary.csv"
            if not summary_csv.exists():
                print(f"  [SKIP] {ev}-{node}: summary CSV not found -> {summary_csv}")
                continue

            df_sum = pd.read_csv(summary_csv)
            if "sample" not in df_sum.columns:
                print(f"  [SKIP] {ev}-{node}: 'sample' column missing -> {summary_csv}")
                continue

            # Merge LHS and summary (by sample)
            merged = pd.merge(lhs, df_sum, on="sample", how="inner")
            if merged.empty:
                print(f"  [SKIP] {ev}-{node}: merge result is empty")
                continue

            # Select only existing response metrics
            resp_cols = [c for c in RESP_CAND if c in merged.columns]
            if not resp_cols:
                print(f"  [SKIP] {ev}-{node}: no response metrics found")
                continue

            print(f"  [NODE] {node} | Response metrics: {resp_cols}")

            X = merged[param_cols]

            all_res = []
            for resp in resp_cols:
                y = merged[resp]
                try:
                    df_prcc = compute_prcc_matrix(X, y)
                    df_prcc.insert(0, "response", resp)
                    df_prcc.insert(0, "node", node)
                    df_prcc.insert(0, "event", ev)
                    all_res.append(df_prcc)
                except Exception as e:
                    print(f"    [WARN] {ev}-{node}-{resp}: PRCC computation failed - {e}")

            if not all_res:
                print(f"  [SKIP] {ev}-{node}: no PRCC results")
                continue

            df_out = pd.concat(all_res, ignore_index=True)

            # Output folder: PRCC/<EVENT>/
            out_ev_dir = PRCC_DIR / ev
            out_ev_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_ev_dir / f"{node}_PRCC.csv"
            df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

            print(f"  [SAVE] {ev}-{node}: PRCC → {out_csv} (rows: {len(df_out)})")

    print("\n[DONE] PRCC computation completed for all events/nodes")
    print(f"       Result root directory: {PRCC_DIR}")


if __name__ == "__main__":
    main()
