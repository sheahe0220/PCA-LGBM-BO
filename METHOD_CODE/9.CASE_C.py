# -*- coding: utf-8 -*-

from __future__ import annotations
import os, time, json, re, shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pickle

from pyswmm import Simulation
from pyswmm.output import Output

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm

import signal
import warnings
warnings.filterwarnings("ignore")

# ===== Path Configuration =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

TEMPLATE_DIR = BASE / "COMPARE" / "5_" / "SWMM"
OBS_PC_DIR   = BASE / "PCA" / "obs_pc_1"
PCA_MODEL_DIR = BASE / "PCA" / "PCA_models_1"

RESULT_ROOT = BASE / "COMPARE" / "5_"
TEMP_DIR    = RESULT_ROOT / "temp"

EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

# ===== Exclusion Targets =====
EXCLUDE_SUBCATCHMENTS = {"MH4306#1"}  # Subcatchments
EXCLUDE_CONDUITS = {  # Conduits
    "L335", "L454", "L456.1", "L456", "L454.1",
    "L453", "L453.2", "L453.1", "L453.1.2", "L477", "L326"
}

# ===== BO Configuration =====
FEATURES = ["Imperv_scale", "Width_scale", "Nimp_scale", "n_pipe_scale"]

# θ search bounds (scale parameters)
BOUNDS = [
    (0.7, 1.2),   # Imperv_scale
    (0.5, 1.5),   # Width_scale
    (0.8, 1.2),   # Nimp_scale
    (0.7, 1.3),   # n_pipe_scale
]

# ===== CASE 1 Initial Values (from Case A) =====
CASE1_INIT = {
    "Imperv_scale": 1.020130854003836,
    "Width_scale": 1.185595882241798,
    "Nimp_scale": 0.9605749788937316,
    "n_pipe_scale": 0.7256634387915548
}

N_INIT = 0           # Initial LHS samples
N_ITER = 200         # Maximum BO iterations
PATIENCE = 999       # Early stopping: iterations without improvement
MIN_IMPROVEMENT = 0.001  # Minimum improvement ratio (0.1%)

N_WORKERS = 4        # Parallel workers for event execution
RNG = np.random.RandomState(42)
AUTOSAVE_EVERY = 10  # Auto-save every N iterations (0=disabled)

# ===== Preprocessing Configuration =====
N_POINTS = 480       # PCA input length
BASELINE_FRAC = 0.10 # Baseline: median of first 10%
CLIP_NEGATIVE = True

# ===== Cost Tracker =====
class CostTracker:
    def __init__(self):
        self.total_start = None
        self.swmm_time = 0.0
        self.gp_fit_time = 0.0
        self.ei_opt_time = 0.0
        self.pca_time = 0.0
        self.n_swmm_calls = 0
        self.n_gp_fits = 0
        self.n_ei_opts = 0
        self.converged_iter = None
        self.init_start = None
        self.init_end = None
        self.best_found_at = None
        self.best_found_time = None

    def start(self):
        self.total_start = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.total_start if self.total_start else 0.0

    def mark_init_start(self):
        self.init_start = time.perf_counter()

    def mark_init_end(self):
        self.init_end = time.perf_counter()

    def mark_best_found(self, iter_num):
        if self.best_found_at is None or iter_num > self.best_found_at:
            self.best_found_at = iter_num
            self.best_found_time = self.elapsed()

    def get_init_time(self):
        if self.init_start and self.init_end:
            return self.init_end - self.init_start
        return 0.0

    def to_dict(self):
        return {
            "total_time_sec": self.elapsed(),
            "init_time_sec": self.get_init_time(),
            "init_to_best_time_sec": self.best_found_time if self.best_found_time else 0.0,
            "swmm_time_sec": self.swmm_time,
            "gp_fit_time_sec": self.gp_fit_time,
            "ei_opt_time_sec": self.ei_opt_time,
            "pca_time_sec": self.pca_time,
            "n_swmm_calls": self.n_swmm_calls,
            "n_gp_fits": self.n_gp_fits,
            "n_ei_opts": self.n_ei_opts,
            "converged_iter": self.converged_iter,
            "best_found_at_iter": self.best_found_at,
            "n_init": N_INIT,
            "n_iter_max": N_ITER,
            "avg_swmm_per_call_sec": self.swmm_time / max(1, self.n_swmm_calls),
            "avg_gp_fit_sec": self.gp_fit_time / max(1, self.n_gp_fits),
            "avg_ei_opt_sec": self.ei_opt_time / max(1, self.n_ei_opts),
        }


COST = CostTracker()

# ===== Global Variables for Checkpointing =====
_g_history = []
_g_result_dir = None
_g_best_dir = None
_g_theta_best = None
_g_J_best = None
_g_interrupted = False

# ===== Utility Functions =====
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_checkpoint(reason: str = "checkpoint"):
    """Save intermediate results"""
    if not _g_history or _g_result_dir is None:
        return
    
    log(f"\n[SAVE] {reason} - Saving current results...")
    
    try:
        # history.csv
        df_hist = pd.DataFrame(_g_history)
        df_hist.to_csv(_g_result_dir / "history.csv", index=False)
        
        # cost_summary.json
        cost_data = COST.to_dict()
        cost_data["save_reason"] = reason
        cost_data["total_iters_saved"] = len(_g_history)
        with open(_g_result_dir / "cost_summary.json", "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        
        # best_result.json
        if _g_theta_best is not None:
            best_payload = {
                "initial_theta_case1": CASE1_INIT,
                "theta_star": {feat: float(_g_theta_best[i]) for i, feat in enumerate(FEATURES)},
                "J_star": float(_g_J_best),
                "n_init_case1": 1,
                "n_init_lhs": N_INIT,
                "n_init_total": 1 + N_INIT,
                "n_iter_completed": len(_g_history) - (1 + N_INIT),
                "total_iters": len(_g_history),
                "bounds": {feat: {"lo": BOUNDS[i][0], "hi": BOUNDS[i][1]} for i, feat in enumerate(FEATURES)},
                "interrupted": _g_interrupted,
            }
            with open(_g_result_dir / "best_result.json", "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, ensure_ascii=False)
        
        log(f"  ✓ Save complete: iter {len(_g_history)}")
    except Exception as e:
        log(f"  [WARN] Save failed: {e}")

def signal_handler(signum, frame):
    """Detect Ctrl+C"""
    global _g_interrupted
    log("\n\n[INTERRUPT] Ctrl+C detected - Saving results and exiting...")
    _g_interrupted = True
    save_checkpoint("interrupted")
    log("[EXIT] Program terminated")
    exit(0)

def clean_temp():
    """Clean temporary folder"""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def split_keep_ws(line: str):
    """Split tokens while preserving whitespace"""
    return re.split(r"(\s+)", line.rstrip("\n"))

def join_keep_ws(parts):
    """Join list split by split_keep_ws"""
    return "".join(parts) + "\n"

def data_idxs(sec: list[str]):
    """Return indices of actual data rows in section (excluding comments/blanks)"""
    out = []
    for i, l in enumerate(sec):
        t = l.strip()
        if (not t) or t.startswith(";;") or t.startswith("["):
            continue
        out.append(i)
    return out

def token_at(parts, idx: int):
    """Return (value, position) by token index"""
    pos = 2 * idx
    if pos >= len(parts):
        return None, None
    return parts[pos], pos

def replace_token_at(parts, idx: int, new_val: str):
    """Replace token at index"""
    pos = 2 * idx
    if pos >= len(parts):
        return
    parts[pos] = new_val

def read_inp(path: Path) -> list[str]:
    """Read INP file as list of lines"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()

def write_inp(path: Path, lines: list[str]):
    """Write lines to INP file"""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.writelines(lines)

def find_section_bounds(lines: list[str], sec_name: str):
    """Find start/end indices of section [sec_name]"""
    start = None
    for i, l in enumerate(lines):
        if l.strip().upper() == f"[{sec_name.upper()}]":
            start = i
            break
    if start is None:
        return None, None
    
    for i in range(start + 1, len(lines)):
        t = lines[i].strip()
        if t.startswith("["):
            return start, i
    return start, len(lines)

def apply_theta_to_inp(lines: list[str], theta: dict) -> list[str]:
    """
    Apply scale parameters to INP file
    theta: {
        "Imperv_scale": float,
        "Width_scale": float,
        "Nimp_scale": float,
        "n_pipe_scale": float
    }
    """
    out = lines.copy()
    
    # 1) SUBCATCHMENTS section
    st, ed = find_section_bounds(out, "SUBCATCHMENTS")
    if st is not None:
        sec = out[st:ed]
        for i in data_idxs(sec):
            line = sec[i]
            parts = split_keep_ws(line)
            
            # subcatchment name (token 0)
            subc_name, _ = token_at(parts, 0)
            if subc_name and subc_name in EXCLUDE_SUBCATCHMENTS:
                continue
            
            # %Imperv (token 3), Width (token 4)
            imperv_str, imperv_pos = token_at(parts, 3)
            width_str, width_pos = token_at(parts, 4)
            
            if imperv_str and imperv_pos is not None:
                try:
                    val = float(imperv_str) * theta.get("Imperv_scale", 1.0)
                    val = max(0.0, min(val, 100.0))  # clamp [0, 100]
                    parts[imperv_pos] = f"{val:.2f}"
                except:
                    pass
            
            if width_str and width_pos is not None:
                try:
                    val = float(width_str) * theta.get("Width_scale", 1.0)
                    val = max(0.01, val)
                    parts[width_pos] = f"{val:.2f}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        
        out[st:ed] = sec
    
    # 2) SUBAREAS section
    st, ed = find_section_bounds(out, "SUBAREAS")
    if st is not None:
        sec = out[st:ed]
        for i in data_idxs(sec):
            line = sec[i]
            parts = split_keep_ws(line)
            
            subc_name, _ = token_at(parts, 0)
            if subc_name and subc_name in EXCLUDE_SUBCATCHMENTS:
                continue
            
            # N-Imperv (token 1)
            nimp_str, nimp_pos = token_at(parts, 1)
            if nimp_str and nimp_pos is not None:
                try:
                    val = float(nimp_str) * theta.get("Nimp_scale", 1.0)
                    val = max(0.001, val)
                    parts[nimp_pos] = f"{val:.4f}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        
        out[st:ed] = sec
    
    # 3) CONDUITS section
    st, ed = find_section_bounds(out, "CONDUITS")
    if st is not None:
        sec = out[st:ed]
        for i in data_idxs(sec):
            line = sec[i]
            parts = split_keep_ws(line)
            
            cond_name, _ = token_at(parts, 0)
            if cond_name and cond_name in EXCLUDE_CONDUITS:
                continue
            
            # Roughness (token 5)
            rough_str, rough_pos = token_at(parts, 5)
            if rough_str and rough_pos is not None:
                try:
                    val = float(rough_str) * theta.get("n_pipe_scale", 1.0)
                    val = max(0.001, val)
                    parts[rough_pos] = f"{val:.4f}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        
        out[st:ed] = sec
    
    return out

def load_pca_models():
    """Load PCA models for each node"""
    models = {}
    for node in NODES:
        node_dir = PCA_MODEL_DIR / node
        pca_path = node_dir / "pca.pkl"
        scl_path = node_dir / "scaler.pkl"
        
        if not pca_path.exists() or not scl_path.exists():
            raise FileNotFoundError(f"[ERR] PCA model not found: {node_dir}")
        
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(scl_path, "rb") as f:
            scaler = pickle.load(f)
        
        models[node] = {"pca": pca, "scaler": scaler}
    
    log(f"[INFO] Loaded PCA models for {len(models)} nodes")
    return models

def load_obs_pcs():
    """Load observed PC scores for each event"""
    obs = {}
    for ev in EVENTS:
        csv_path = OBS_PC_DIR / f"obs_pc_{ev}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"[ERR] Observed PC file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        obs[ev] = df.iloc[0].to_dict()
    
    log(f"[INFO] Loaded observed PC scores for {len(obs)} events")
    return obs

def compute_evr_weights(pca_models):
    """
    Compute EVR weights for each PC
    Returns: dict[node][PC_name] = weight
    """
    weights = {}
    for node, model_dict in pca_models.items():
        pca = model_dict["pca"]
        evr = pca.explained_variance_ratio_
        n_pc = len(evr)
        
        weights[node] = {}
        for i in range(n_pc):
            pc_name = f"{node}_PC{i+1}"
            weights[node][pc_name] = float(evr[i])
    
    log("[INFO] Computed EVR weights")
    return weights

def preprocess_series(x_raw: np.ndarray, n_target: int) -> np.ndarray:
    """
    Preprocess 1D time series to match LHS-PCA input format:
      1) NaN handling
      2) Baseline removal (median of first 10%)
      3) Negative clipping (optional)
      4) Resample to n_target length
    """
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
    
    # 2) Baseline (median of first 10%)
    head_n = max(1, int(L * BASELINE_FRAC))
    base_region = x[:head_n]
    if np.any(~np.isnan(base_region)):
        baseline = float(np.nanmedian(base_region))
    else:
        baseline = float(np.nanmedian(x))
    
    x = x - baseline
    
    # 3) Negative clipping
    if CLIP_NEGATIVE:
        x = np.maximum(x, 0.0)
    
    # 4) Resample to n_target length
    if L == n_target:
        return x
    
    orig_pos = np.linspace(0.0, 1.0, L)
    new_pos  = np.linspace(0.0, 1.0, n_target)
    x_resampled = np.interp(new_pos, orig_pos, x)
    
    return x_resampled

def extract_depth_from_out(out_path: Path, node_id: str) -> np.ndarray:
    """Extract depth time series from SWMM output file"""
    try:
        with Output(str(out_path)) as out:
            series = out.node_series(node_id, "DEPTH")
            
            # Convert to DataFrame
            if isinstance(series, dict):
                s = pd.Series(series)
                s.index = pd.to_datetime(s.index)
                s = s.sort_index().astype(float)
            elif isinstance(series, pd.Series):
                s = series.copy()
                s.index = pd.to_datetime(s.index)
                s = s.sort_index().astype(float)
            else:
                df = pd.DataFrame(list(series), columns=["t", "value"])
                df["t"] = pd.to_datetime(df["t"])
                df = df.sort_values("t")
                s = df.set_index("t")["value"]
            
            return s.values
    except Exception as e:
        raise RuntimeError(f"Failed to extract depth from {out_path} for {node_id}: {e}")

def run_swmm_parallel(theta: dict, run_id: str):
    """
    Run SWMM simulations for all events in parallel
    Returns: dict[event] = out_path
    """
    t0 = time.perf_counter()
    
    def run_single_event(event: str):
        # Apply theta to template
        template_path = TEMPLATE_DIR / f"10mm_{event}.inp"
        template_text = read_inp(template_path)
        modified_text = apply_theta_to_inp(template_text, theta)
        
        # Write modified INP
        temp_inp = TEMP_DIR / f"{run_id}_{event}.inp"
        write_inp(temp_inp, modified_text)
        
        # Run SWMM
        temp_out = TEMP_DIR / f"{run_id}_{event}.out"
        temp_rpt = TEMP_DIR / f"{run_id}_{event}.rpt"
        
        with Simulation(str(temp_inp), reportfile=str(temp_rpt), outputfile=str(temp_out)) as sim:
            for _ in sim:
                pass
        
        return event, temp_out
    
    # Parallel execution
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(run_single_event, ev): ev for ev in EVENTS}
        
        for fut in as_completed(futures):
            try:
                ev, out_path = fut.result()
                results[ev] = out_path
            except Exception as e:
                ev = futures[fut]
                raise RuntimeError(f"[ERROR] SWMM failed for {ev}: {e}")
    
    elapsed = time.perf_counter() - t0
    COST.swmm_time += elapsed
    COST.n_swmm_calls += len(EVENTS)
    
    return results

def compute_pca_scores(depths: dict, pca_models):
    """
    Compute PCA scores from depth time series
    
    Args:
        depths: dict[event][node] = np.ndarray (raw depth series)
        pca_models: dict[node] = {"pca": PCA, "scaler": StandardScaler}
    
    Returns:
        dict[event][node_PC] = score
    """
    t0 = time.perf_counter()
    
    scores = {}
    for ev in EVENTS:
        scores[ev] = {}
        
        for node in NODES:
            # Preprocess
            x_raw = depths[ev][node]
            x_feat = preprocess_series(x_raw, N_POINTS)
            
            # Scale + PCA
            scaler = pca_models[node]["scaler"]
            pca = pca_models[node]["pca"]
            
            x_feat_2d = x_feat.reshape(1, -1)
            x_scaled = scaler.transform(x_feat_2d)
            x_pc = pca.transform(x_scaled)
            
            n_pc = x_pc.shape[1]
            for i in range(n_pc):
                pc_name = f"{node}_PC{i+1}"
                scores[ev][pc_name] = float(x_pc[0, i])
    
    COST.pca_time += time.perf_counter() - t0
    
    return scores

def compute_objective(sim_pcs: dict, obs_pcs: dict, evr_weights: dict):
    """
    Compute EVR-weighted RMSE objective function
    
    Args:
        sim_pcs: dict[event][node_PC] = score
        obs_pcs: dict[event][node_PC] = score
        evr_weights: dict[node][node_PC] = weight
    
    Returns:
        J_total: float (weighted RMSE across all events)
        J_events: dict[event] = RMSE
    """
    J_events = {}
    
    for ev in EVENTS:
        sim = sim_pcs[ev]
        obs = obs_pcs[ev]
        
        squared_errors = []
        weights = []
        
        for pc_name in sim.keys():
            if pc_name not in obs:
                continue
            
            # Find node
            node = pc_name.split("_")[0]
            if node not in evr_weights or pc_name not in evr_weights[node]:
                continue
            
            w = evr_weights[node][pc_name]
            err = sim[pc_name] - obs[pc_name]
            
            squared_errors.append(err ** 2)
            weights.append(w)
        
        if not squared_errors:
            J_events[ev] = 999.0
        else:
            mse = np.average(squared_errors, weights=weights)
            J_events[ev] = np.sqrt(mse)
    
    # Total objective: average RMSE across events
    J_total = np.mean(list(J_events.values()))
    
    return J_total, J_events

def evaluate_theta(theta: dict, run_id: str, pca_models, obs_pcs, evr_weights):
    """
    Evaluate theta by running SWMM and computing objective function
    
    Returns:
        J_total: float
        J_events: dict[event] = float
    """
    # 1) Run SWMM for all events
    out_files = run_swmm_parallel(theta, run_id)
    
    # 2) Extract depths
    depths = {}
    for ev, out_path in out_files.items():
        depths[ev] = {}
        for node in NODES:
            depths[ev][node] = extract_depth_from_out(out_path, node)
    
    # 3) Compute PCA scores
    sim_pcs = compute_pca_scores(depths, pca_models)
    
    # 4) Compute objective
    J_total, J_events = compute_objective(sim_pcs, obs_pcs, evr_weights)
    
    return J_total, J_events

def make_gp_kernel(n_dim: int):
    """Create GP kernel"""
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=[1.0] * n_dim, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    )

def expected_improvement(mu, sigma, y_best, xi=0.01):
    """Expected Improvement acquisition function"""
    mu = mu.ravel()
    sigma = sigma.ravel()
    
    with np.errstate(divide='warn'):
        imp = y_best - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    
    return ei

def main():
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize
    COST.start()
    clean_temp()
    
    # Setup result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ensure_dir(RESULT_ROOT / f"result_{timestamp}")
    best_dir = ensure_dir(result_dir / "best_inp")
    
    global _g_result_dir, _g_best_dir
    _g_result_dir = result_dir
    _g_best_dir = best_dir
    
    log("=" * 80)
    log("Case B: GP-BO with Warm Start (Case 1 initialization)")
    log("=" * 80)
    log(f"Result directory: {result_dir}")
    log(f"Initial theta (Case 1): {CASE1_INIT}")
    log(f"N_INIT (LHS): {N_INIT}")
    log(f"N_ITER (BO): {N_ITER}")
    log(f"PATIENCE: {PATIENCE}")
    log(f"MIN_IMPROVEMENT: {MIN_IMPROVEMENT}")
    log(f"AUTOSAVE_EVERY: {AUTOSAVE_EVERY}")
    log("=" * 80)
    
    # Load models
    log("\n[LOAD] Loading PCA models and observed PC scores...")
    pca_models = load_pca_models()
    obs_pcs = load_obs_pcs()
    evr_weights = compute_evr_weights(pca_models)
    
    # Optimization loop
    try:
        history = []
        
        # Initialize with Case 1 + LHS
        COST.mark_init_start()
        log(f"\n[INIT] Initializing with {1 + N_INIT} samples...")
        
        X_list = []
        y_list = []
        
        # Case 1 initial value
        X_init = [np.array([CASE1_INIT[feat] for feat in FEATURES])]
        
        # LHS samples (if N_INIT > 0)
        if N_INIT > 0:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=len(FEATURES), seed=RNG)
            lhs_samples = sampler.random(n=N_INIT)
            
            for i in range(N_INIT):
                theta_arr = np.array([
                    BOUNDS[j][0] + lhs_samples[i, j] * (BOUNDS[j][1] - BOUNDS[j][0])
                    for j in range(len(FEATURES))
                ])
                X_init.append(theta_arr)
        
        # Evaluate initial samples
        for i in range(len(X_init)):
            iter_start_time = time.perf_counter()
            theta_arr = X_init[i]
            theta_dict = {feat: float(val) for feat, val in zip(FEATURES, theta_arr)}
            
            log(f"\n[INIT] Sample {i+1}/{N_INIT+1}" + (" (Case 1)" if i == 0 else f" (LHS {i})"))
            
            try:
                J_total, J_events = evaluate_theta(theta_dict, f"init_{i:03d}", pca_models, obs_pcs, evr_weights)
            except Exception as e:
                log(f"  [ERROR] Evaluation failed: {e}")
                J_total = 999.0
                J_events = {ev: 999.0 for ev in EVENTS}
            
            X_list.append(theta_arr)
            y_list.append(J_total)
            
            iter_time = time.perf_counter() - iter_start_time
            hist_entry = {
                "iter": i + 1,
                "stage": "INIT_CASE1" if i == 0 else "INIT_LHS",
                **{f"theta_{feat}": theta_dict[feat] for feat in FEATURES},
                **{f"J_{ev}": J_events[ev] for ev in EVENTS},
                "J_total": J_total,
                "is_best": 0,
                "iter_time_sec": iter_time,
                "elapsed_time_sec": COST.elapsed(),
            }
            history.append(hist_entry)
            _g_history.append(hist_entry)
            
            # Auto-save
            if AUTOSAVE_EVERY > 0 and (i + 1) % AUTOSAVE_EVERY == 0:
                save_checkpoint(f"autosave_init_{i+1}")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        COST.mark_init_end()
        
        best_idx = int(np.argmin(y))
        J_best = float(y[best_idx])
        theta_best = X[best_idx].copy()
        
        history[best_idx]["is_best"] = 1
        _g_history[best_idx]["is_best"] = 1
        _g_theta_best = theta_best
        _g_J_best = J_best
        COST.mark_best_found(best_idx + 1)
        
        log(f"\n[INIT] Complete | J_best = {J_best:.6f}")
        log(f"  θ_best = {', '.join([f'{FEATURES[i]}={theta_best[i]:.4f}' for i in range(len(FEATURES))])}")
        
        # BO iterations
        log(f"\n[BO] Starting iterations (max {N_ITER}, patience={PATIENCE})...")
        
        no_improve_count = 0
        
        for it in range(1, N_ITER + 1):
            iter_start_time = time.perf_counter()
            log(f"\n[BO] Iter {it}/{N_ITER}")
            
            # GP training (with cost tracking)
            t0_gp = time.perf_counter()
            kernel = make_gp_kernel(len(FEATURES))
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=3,
                random_state=RNG,
            )
            gp.fit(X, y)
            COST.gp_fit_time += time.perf_counter() - t0_gp
            COST.n_gp_fits += 1
            
            # EI maximization (multi-start, with cost tracking)
            t0_ei = time.perf_counter()
            best_ei = -np.inf
            best_x = None
            
            for _ in range(10):
                x0 = RNG.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
                
                def obj_ei(x):
                    x = np.array(x).reshape(1, -1)
                    mu, sigma = gp.predict(x, return_std=True)
                    ei = expected_improvement(mu, sigma, J_best, xi=0.01)
                    return -ei[0]
                
                res = minimize(obj_ei, x0=x0, bounds=BOUNDS, method="L-BFGS-B")
                
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    best_x = res.x
            
            COST.ei_opt_time += time.perf_counter() - t0_ei
            COST.n_ei_opts += 1
            
            x_next = best_x
            theta_dict = {feat: float(val) for feat, val in zip(FEATURES, x_next)}
            
            # Evaluate θ
            try:
                J_next, J_events = evaluate_theta(theta_dict, f"bo{it:03d}", pca_models, obs_pcs, evr_weights)
            except Exception as e:
                log(f"  [ERROR] Evaluation failed: {e}")
                J_next = 999.0
                J_events = {ev: 999.0 for ev in EVENTS}
            
            # Update
            X = np.vstack([X, x_next])
            y = np.concatenate([y, [J_next]])
            
            is_best = 0
            if J_next < J_best * (1 - MIN_IMPROVEMENT):
                improvement = (J_best - J_next) / J_best * 100
                J_best = J_next
                theta_best = x_next.copy()
                is_best = 1
                no_improve_count = 0
                _g_theta_best = theta_best
                _g_J_best = J_best
                COST.mark_best_found(N_INIT + it)
                
                log(f"  ⬇ NEW BEST! (improvement: {improvement:.2f}%)")
                
                # Save best INP
                for event in EVENTS:
                    template_path = TEMPLATE_DIR / f"10mm_{event}.inp"
                    template_text = read_inp(template_path)
                    modified_text = apply_theta_to_inp(template_text, theta_dict)
                    
                    out_inp = best_dir / f"best_iter{it:03d}_{event}.inp"
                    write_inp(out_inp, modified_text)
            else:
                no_improve_count += 1
            
            iter_time = time.perf_counter() - iter_start_time
            hist_entry = {
                "iter": N_INIT + it,
                "stage": "BO",
                **{f"theta_{feat}": theta_dict[feat] for feat in FEATURES},
                **{f"J_{ev}": J_events[ev] for ev in EVENTS},
                "J_total": J_next,
                "is_best": is_best,
                "iter_time_sec": iter_time,
                "elapsed_time_sec": COST.elapsed(),
            }
            history.append(hist_entry)
            _g_history.append(hist_entry)
            
            # Auto-save
            if AUTOSAVE_EVERY > 0 and (N_INIT + it) % AUTOSAVE_EVERY == 0:
                save_checkpoint(f"autosave_iter_{N_INIT + it}")
            
            log(f"  Current J_best = {J_best:.6f} | No improvement count: {no_improve_count}/{PATIENCE}")
            
            # Early stopping
            if no_improve_count >= PATIENCE:
                log(f"\n[STOP] Early stopping: no improvement for {PATIENCE} iterations")
                COST.converged_iter = N_INIT + it
                break
        
        if COST.converged_iter is None:
            COST.converged_iter = len(history)
    
    except KeyboardInterrupt:
        log("\n[INTERRUPT] Ctrl+C detected")
        save_checkpoint("interrupted")
        raise
    except Exception as e:
        log(f"\n[ERROR] Error occurred: {e}")
        save_checkpoint("error")
        raise
    finally:
        # Always execute: final save
        if len(_g_history) > 0:
            save_checkpoint("final")

    # Save results (final success) - already saved in finally block
    log("\n[SUCCESS] Optimization completed normally")
    
    # Get cost data
    cost_data = COST.to_dict()
    
    # Plot convergence curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        df_hist = pd.DataFrame(history)
        iter_nums = df_hist["iter"].values
        J_totals = df_hist["J_total"].values
        
        # Cumulative minimum
        J_best_curve = np.minimum.accumulate(J_totals)
        
        plt.plot(iter_nums, J_totals, 'o-', alpha=0.5, label='J_total')
        plt.plot(iter_nums, J_best_curve, 'r-', linewidth=2, label='J_best')
        plt.axvline(1, color='gray', linestyle='--', label='INIT → BO')
        
        plt.xlabel('Iteration')
        plt.ylabel('J_total (EVR-weighted RMSE)')
        plt.title('Case B: GP-BO Convergence (Warm Start)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(result_dir / "convergence.png", dpi=150)
        log(f"  - convergence.png saved")
    except Exception as e:
        log(f"  [WARN] Plot generation failed: {e}")
    
    log("\n" + "=" * 80)
    log("Case B optimization complete!")
    log("=" * 80)
    log(f"Optimal parameters:")
    for feat, val in zip(FEATURES, theta_best):
        log(f"  - {feat}: {val:.6f}")
    log(f"Optimal objective: J* = {J_best:.6f}")
    log("\nCost summary:")
    log(f"  Total time: {cost_data['total_time_sec']:.2f} sec ({cost_data['total_time_sec']/60:.1f} min)")
    log(f"  SWMM time: {cost_data['swmm_time_sec']:.2f} sec ({cost_data['swmm_time_sec']/60:.1f} min)")
    log(f"  GP fit time: {cost_data['gp_fit_time_sec']:.2f} sec")
    log(f"  EI optimization time: {cost_data['ei_opt_time_sec']:.2f} sec")
    log(f"  PCA processing time: {cost_data['pca_time_sec']:.2f} sec")
    log(f"  SWMM calls: {cost_data['n_swmm_calls']}")
    log(f"  SWMM avg time: {cost_data['avg_swmm_per_call_sec']:.3f} sec/call")
    log(f"  GP fits: {cost_data['n_gp_fits']}")
    log(f"  Converged iteration: {cost_data['converged_iter']}")
    log(f"\nResult folder: {result_dir}")
    log("=" * 80)


if __name__ == "__main__":
    main()
