# %%
from pathlib import Path
import pandas as pd
import numpy as np

# ===============================
# Configuration
# ===============================
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

IN_BASE  = BASE / "LHS_DEPTH"                 # LHS SWMM results (root)
OUT_BASE = BASE / "PCA" / "PCA_resampled_1"   # Output directory for resampled PCA inputs

EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

N_POINTS        = 480     # Number of resampling points
BASELINE_FRAC   = 0.10    # Baseline estimated from the first 10% segment
CLIP_NEGATIVE   = True    # After baseline removal, clip negatives to 0

OUT_BASE.mkdir(parents=True, exist_ok=True)


def detect_time_depth_cols(df: pd.DataFrame):
    time_candidates  = ["t", "time", "Time", "timestamp", "Datetime", "date_time"]
    depth_candidates = ["Depth", "depth", "value", "Depth_m", "WaterLevel", "water_level"]

    t_col = None
    d_col = None

    for c in time_candidates:
        if c in df.columns:
            t_col = c
            break

    for c in depth_candidates:
        if c in df.columns:
            d_col = c
            break

    if t_col is None:
        raise RuntimeError(f"Time column not found. Candidates: {time_candidates}")
    if d_col is None:
        raise RuntimeError(f"Depth column not found. Candidates: {depth_candidates}")

    return t_col, d_col


def load_and_resample_series(csv_path: Path) -> tuple[np.ndarray, float]:
    df = pd.read_csv(csv_path)
    t_col, d_col = detect_time_depth_cols(df)

    # Parse and sort time column
    # → normalize to numpy.datetime64
    df[t_col] = pd.to_datetime(df[t_col]).astype("datetime64[ns]")
    df = df.sort_values(t_col)

    # Drop duplicate timestamps (keep first)
    df = df.drop_duplicates(subset=t_col, keep="first")

    t = df[t_col].to_numpy()  # dtype=datetime64[ns]
    y = pd.to_numeric(df[d_col], errors="coerce").to_numpy()

    # NaN handling (edge fill + linear interpolation)
    if np.all(np.isnan(y)):
        y = np.zeros_like(y, dtype=float)
    else:
        s = pd.Series(y)
        s = s.interpolate(limit_direction="both")
        y = s.to_numpy()

    n = len(y)
    if n == 0:
        return np.zeros(N_POINTS, dtype=float), 0.0

    # Baseline estimation: median over the first 10% segment
    head_n = max(1, int(n * BASELINE_FRAC))
    baseline_region = y[:head_n]
    if np.any(~np.isnan(baseline_region)):
        baseline = float(np.nanmedian(baseline_region))
    else:
        baseline = float(np.nanmedian(y))

    # Baseline removal
    y = y - baseline
    if CLIP_NEGATIVE:
        y = np.maximum(y, 0.0)

    # Normalize time axis to 0~1 (numpy.timedelta64 compatible)
    t0 = t[0]
    t1 = t[-1]
    if t1 == t0:
        # If all timestamps are identical, use uniform 0~1 spacing
        s = np.linspace(0.0, 1.0, n)
    else:
        # Convert to seconds: timedelta64 / np.timedelta64(1, 's')
        dt_seconds = (t1 - t0) / np.timedelta64(1, "s")
        s = ((t - t0) / np.timedelta64(1, "s")) / dt_seconds

    # Resampling grid
    s_grid = np.linspace(0.0, 1.0, N_POINTS)

    # Interpolation
    y_grid = np.interp(s_grid, s, y)

    return y_grid, baseline


def main():
    for event in EVENTS:
        for node in NODES:
            in_dir  = IN_BASE / event / node
            out_dir = OUT_BASE / event / node
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"[WARN] Input directory not found: {in_dir}")
                continue

            rows = []
            baselines = []

            for csv_path in sorted(in_dir.glob("sample_*.csv")):
                sample_id = csv_path.stem  # e.g., sample_001
                try:
                    y_grid, baseline = load_and_resample_series(csv_path)
                except Exception as e:
                    print(f"[ERR] {event}-{node}-{csv_path.name}: {e}")
                    continue

                row = {"sample": sample_id}
                for i in range(N_POINTS):
                    row[f"f{i+1}"] = float(y_grid[i])
                rows.append(row)
                baselines.append({"sample": sample_id, "baseline": baseline})

            if not rows:
                print(f"[WARN] {event}-{node}: No processed samples.")
                continue

            df_out = pd.DataFrame(rows)
            df_out.to_csv(out_dir / f"PCA_input_{event}_{node}.csv", index=False)

            df_base = pd.DataFrame(baselines)
            df_base.to_csv(out_dir / f"baseline_{event}_{node}.csv", index=False)

            print(f"[OK] {event}-{node}: Resampled {len(rows)} samples -> {out_dir}")

    print("\n[DONE] Baseline + resampling stage finished.")


if __name__ == "__main__":
    main()

# %%
