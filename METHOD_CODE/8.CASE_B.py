# -*- coding: utf-8 -*-
# CASE_2_compute.py - cost tracking revised version

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

# ===== Paths =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

TEMPLATE_DIR = BASE / "COMPARE" / "55_" / "SWMM"
OBS_PC_DIR   = BASE / "PCA" / "obs_pc_1"
PCA_MODEL_DIR = BASE / "PCA" / "PCA_models_1"
LHS_PARAMS_PATH = BASE / "input" / "lhs_samples_4d.csv"
LHS_PC_DIR = BASE / "PCA"

RESULT_ROOT = BASE / "COMPARE" / "55_"
TEMP_DIR    = RESULT_ROOT / "temp"

EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

EXCLUDE_SUBCATCHMENTS = {"MH4306#1"}
EXCLUDE_CONDUITS = {
    "L335", "L454", "L456.1", "L456", "L454.1",
    "L453", "L453.2", "L453.1", "L453.1.2", "L477", "L326"
}

# ===== BO settings =====
FEATURES = ["Imperv_scale", "Width_scale", "Nimp_scale", "n_pipe_scale"]

BOUNDS = [
    (0.7, 1.2),
    (0.5, 1.5),
    (0.8, 1.2),
    (0.7, 1.3),
]

N_INIT = 0
N_ITER = 200
PATIENCE = 999
MIN_IMPROVEMENT = 0.001

N_WORKERS = 4
RNG = np.random.RandomState(42)
AUTOSAVE_EVERY = 10  # Auto-save every N iterations (0=disabled)

N_POINTS = 480
BASELINE_FRAC = 0.10
CLIP_NEGATIVE = True


# ===== Cost tracking - revised version =====
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

def load_random_lhs_init(pca_models, obs_pcs, evr_weights, seed=42):
    log(f"\n[INIT] Randomly select 1 from LHS 300 (seed={seed})...")
    lhs_df = pd.read_csv(LHS_PARAMS_PATH)
    np.random.seed(seed)
    idx = np.random.randint(0, len(lhs_df))
    row = lhs_df.iloc[idx]
    theta = {"Imperv_scale": float(row["Imperv"]), "Width_scale": float(row["Width"]),
             "Nimp_scale": float(row["Nimp"]), "n_pipe_scale": float(row["n_pipe"])}
    log(f"  ✓ sample_{idx+1:03d}: [{theta['Imperv_scale']:.4f}, {theta['Width_scale']:.4f}, {theta['Nimp_scale']:.4f}, {theta['n_pipe_scale']:.4f}]")
    J_events = {}
    for event in EVENTS:
        y_df = pd.read_csv(LHS_PC_DIR / f"Y_{event}_selected.csv")
        y_row = y_df.iloc[idx]
        obs_dict = obs_pcs[event]
        rmse_sum = 0.0
        weight_sum = 0.0
        for node in NODES:
            for i in range(1, 5):
                pc_name = f"{node}_PC{i}"
                pc_model = y_row[pc_name]
                pc_obs = obs_dict[pc_name]
                w = evr_weights.get(pc_name, 1.0)
                rmse_sum += w * (pc_model - pc_obs)**2
                weight_sum += w
        J_events[event] = float(np.sqrt(rmse_sum / weight_sum))
    J_total = float(np.mean(list(J_events.values())))
    log(f"  ✓ J_total={J_total:.6f}")
    X = np.array([[theta["Imperv_scale"], theta["Width_scale"], theta["Nimp_scale"], theta["n_pipe_scale"]]])
    y = np.array([J_total])
    hist = {"iter": 1, "stage": "INIT_LHS", "theta_Imperv_scale": theta["Imperv_scale"],
            "theta_Width_scale": theta["Width_scale"], "theta_Nimp_scale": theta["Nimp_scale"],
            "theta_n_pipe_scale": theta["n_pipe_scale"], "J_R1": J_events["R1"], "J_R2": J_events["R2"],
            "J_R5": J_events["R5"], "J_R6": J_events["R6"], "J_total": J_total, "is_best": 1,
            "iter_time_sec": 0.0, "elapsed_time_sec": 0.0}
    return X, y, hist



# ===== Global variables for checkpointing =====
_g_history = []
_g_result_dir = None
_g_best_dir = None
_g_theta_best = None
_g_J_best = None
_g_interrupted = False


# ===== Utils =====
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_checkpoint(reason: str = "checkpoint"):
    """Save intermediate results"""
    if not _g_history or _g_result_dir is None:
        return
    
    log(f"\n[SAVE] {reason} - Saving results so far...")
    
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
        
        # best_result.json (best so far)
        if _g_theta_best is not None:
            best_payload = {
                "theta_star": {feat: float(_g_theta_best[i]) for i, feat in enumerate(FEATURES)},
                "J_star": float(_g_J_best),
                "n_init": N_INIT,
                "n_iter_completed": len(_g_history) - N_INIT,
                "total_iters": len(_g_history),
                "bounds": {feat: {"lo": BOUNDS[i][0], "hi": BOUNDS[i][1]} for i, feat in enumerate(FEATURES)},
                "interrupted": _g_interrupted,
            }
            with open(_g_result_dir / "best_result.json", "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, ensure_ascii=False)
        
        log(f"  ✓ Saved: iter {len(_g_history)}")
    except Exception as e:
        log(f"  [WARN] Save failed: {e}")

def signal_handler(signum, frame):
    """Detect Ctrl+C"""
    global _g_interrupted
    log("\n\n[INTERRUPT] Ctrl+C detected - saving results and exiting...")
    _g_interrupted = True
    save_checkpoint("interrupted")
    log("[EXIT] Program terminated")
    exit(0)

def clean_temp():
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
            time.sleep(0.3)
        except:
            pass
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def safe_remove_file(fpath: Path):
    try:
        if fpath.exists():
            fpath.unlink()
    except:
        pass


# ===== LHS =====
def latin_hypercube(n_samples: int, n_dim: int, rng) -> np.ndarray:
    U = np.zeros((n_samples, n_dim))
    for j in range(n_dim):
        perm = rng.permutation(n_samples)
        U[:, j] = (perm + rng.rand(n_samples)) / n_samples
    return U

def scale_to_bounds(U: np.ndarray, bounds: list) -> np.ndarray:
    out = np.zeros_like(U)
    for i, (lo, hi) in enumerate(bounds):
        out[:, i] = lo + (hi - lo) * U[:, i]
    return out


# ===== INP processing =====
def read_inp(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_inp(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def split_keep_ws(line: str):
    return re.split(r'(\s+)', line.rstrip("\n"))

def join_keep_ws(parts) -> str:
    return "".join(parts)

def token_at(parts, idx: int):
    pos = 2 * idx
    if pos >= len(parts):
        return None, None
    return parts[pos], pos

def parse_section(lines, name: str):
    s = None
    for i, l in enumerate(lines):
        if l.strip().upper() == f"[{name.upper()}]":
            s = i
            break
    if s is None:
        return None, None
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if re.match(r"\s*\[.+\]\s*$", lines[j]):
            e = j
            break
    return s, e

def data_idxs(sec_lines):
    out = []
    for i in range(1, len(sec_lines)):
        s = sec_lines[i].strip()
        if (not s) or s.startswith(";"):
            continue
        out.append(i)
    return out

def apply_theta_to_inp(template_text: str, theta: dict) -> str:
    lines = template_text.splitlines()
    
    imp_sc = theta.get("Imperv_scale", 1.0)
    wid_sc = theta.get("Width_scale", 1.0)
    nimp_sc = theta.get("Nimp_scale", 1.0)
    npipe_sc = theta.get("n_pipe_scale", 1.0)
    
    # SUBCATCHMENTS
    s, e = parse_section(lines, "SUBCATCHMENTS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name_tok, _ = token_at(parts, 0)
            name = (name_tok or "").strip()
            if name in EXCLUDE_SUBCATCHMENTS:
                continue
            
            imp_tok, imp_pos = token_at(parts, 4)
            if imp_tok and imp_pos:
                try:
                    base_imp = float(imp_tok)
                    new_imp = max(0.0, min(base_imp * imp_sc, 100.0))
                    parts[imp_pos] = f"{new_imp:.6g}"
                except:
                    pass
            
            wid_tok, wid_pos = token_at(parts, 5)
            if wid_tok and wid_pos:
                try:
                    base_wid = float(wid_tok)
                    new_wid = base_wid * wid_sc
                    parts[wid_pos] = f"{new_wid:.6g}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec
    
    # SUBAREAS
    s, e = parse_section(lines, "SUBAREAS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name_tok, _ = token_at(parts, 0)
            name = (name_tok or "").strip()
            if name in EXCLUDE_SUBCATCHMENTS:
                continue
            
            nimp_tok, nimp_pos = token_at(parts, 1)
            if nimp_tok and nimp_pos:
                try:
                    base_nimp = float(nimp_tok)
                    new_nimp = base_nimp * nimp_sc
                    parts[nimp_pos] = f"{new_nimp:.6g}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec
    
    # CONDUITS
    s, e = parse_section(lines, "CONDUITS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name_tok, _ = token_at(parts, 0)
            name = (name_tok or "").strip()
            if name in EXCLUDE_CONDUITS:
                continue
            
            rough_tok, rough_pos = token_at(parts, 4)
            if rough_tok and rough_pos:
                try:
                    base_n = float(rough_tok)
                    new_n = base_n * npipe_sc
                    parts[rough_pos] = f"{new_n:.6g}"
                except:
                    pass
            
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec
    
    return "\n".join(lines)


# ===== SWMM execution =====
def run_swmm_event(event: str, theta: dict, iter_id: str):
    template_path = TEMPLATE_DIR / f"10mm_{event}.inp"
    if not template_path.exists():
        raise FileNotFoundError(f"Template INP not found: {template_path}")
    
    template_text = read_inp(template_path)
    modified_text = apply_theta_to_inp(template_text, theta)
    
    inp_path = TEMP_DIR / f"{iter_id}_{event}.inp"
    out_path = TEMP_DIR / f"{iter_id}_{event}.out"
    rpt_path = TEMP_DIR / f"{iter_id}_{event}.rpt"
    
    write_inp(inp_path, modified_text)
    
    try:
        with Simulation(str(inp_path), reportfile=str(rpt_path), outputfile=str(out_path)) as sim:
            for _ in sim:
                pass
    except Exception as e:
        raise RuntimeError(f"SWMM execution failed ({event}): {e}")
    
    try:
        from pyswmm.output import NodeAttribute
    except:
        NodeAttribute = None
    
    depth_data = {}
    with Output(str(out_path)) as out:
        for node in NODES:
            if NodeAttribute:
                data = out.node_series(node, NodeAttribute.DEPTH)
            else:
                data = out.node_series(node, "DEPTH")
            
            if isinstance(data, dict):
                s = pd.Series(data)
                s.index = pd.to_datetime(s.index)
                s = s.sort_index().astype(float)
                df = s.rename("depth").to_frame()
                df["time"] = df.index
                depth_data[node] = df[["time", "depth"]].reset_index(drop=True)
            else:
                df = pd.DataFrame(list(data), columns=["time", "depth"])
                df["time"] = pd.to_datetime(df["time"])
                depth_data[node] = df.sort_values("time").reset_index(drop=True)
    
    for f in [inp_path, out_path, rpt_path]:
        safe_remove_file(f)
    
    return event, depth_data


# ===== Water-level preprocessing + PCA =====
def preprocess_depth_series(df: pd.DataFrame) -> np.ndarray:
    x = df["depth"].to_numpy(dtype=float)
    
    if np.all(np.isnan(x)):
        x = np.zeros_like(x, dtype=float)
    else:
        s = pd.Series(x)
        s = s.interpolate(limit_direction="both")
        x = s.to_numpy()
    
    L = len(x)
    if L == 0:
        return np.zeros(N_POINTS, dtype=float)
    
    head_n = max(1, int(L * BASELINE_FRAC))
    baseline_region = x[:head_n]
    if np.any(~np.isnan(baseline_region)):
        baseline = float(np.nanmedian(baseline_region))
    else:
        baseline = float(np.nanmedian(x))
    
    x = x - baseline
    
    if CLIP_NEGATIVE:
        x = np.maximum(x, 0.0)
    
    if L == N_POINTS:
        return x
    
    orig_pos = np.linspace(0.0, 1.0, L)
    new_pos = np.linspace(0.0, 1.0, N_POINTS)
    x_resampled = np.interp(new_pos, orig_pos, x)
    
    return x_resampled


def load_pca_models() -> dict:
    """Load PCA models {node: (scaler, pca)}"""
    models = {}
    for node in NODES:
        scaler_path = PCA_MODEL_DIR / node / "scaler.pkl"
        pca_path = PCA_MODEL_DIR / node / "pca.pkl"
        
        if not scaler_path.exists() or not pca_path.exists():
            raise FileNotFoundError(f"PCA model not found: {node}")
        
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        
        models[node] = (scaler, pca)
    
    return models


def load_obs_pcs() -> dict:
    obs_pcs = {}
    for event in EVENTS:
        path = OBS_PC_DIR / f"obs_pc_{event}.csv"
        df = pd.read_csv(path)
        obs_pcs[event] = df.iloc[0].to_dict()
    return obs_pcs


def load_evr_weights() -> dict:
    """Load EVR weights {node_PC: evr}"""
    evr_path = BASE / "PCA" / "PC_selected_evr.csv"
    if not evr_path.exists():
        log(f"EVR file not found, using weight=1: {evr_path}")
        return {}
    
    df = pd.read_csv(evr_path)
    weights = {}
    for _, row in df.iterrows():
        target = row["target"]  # e.g., M0113_PC1
        evr = float(row["evr"])
        weights[target] = evr
    
    return weights

# ===== Objective function - cost tracking revised =====
def compute_weighted_rmse(model_pcs: dict, obs_pcs: dict, evr_weights: dict) -> float:
    common_keys = set(model_pcs.keys()) & set(obs_pcs.keys())
    if not common_keys:
        return 999.0
    
    w = np.array([evr_weights.get(k, 1.0) for k in common_keys])
    diff = np.array([model_pcs[k] - obs_pcs[k] for k in common_keys])
    
    rmse = np.sqrt(np.sum(w * diff**2) / np.sum(w))
    return float(rmse)


def evaluate_theta(theta_dict: dict, iter_id: str, pca_models: dict, obs_pcs: dict, evr_weights: dict):
    log(f"  θ = {theta_dict}")
    
    J_events = {}
    
    # Measure SWMM execution time (revised)
    t0_swmm = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {
            exe.submit(run_swmm_event, event, theta_dict, iter_id): event
            for event in EVENTS
        }
        
        results = {}
        for fut in as_completed(futures):
            event = futures[fut]
            ev, depth_data = fut.result()
            results[ev] = depth_data
            log(f"    ├─ {ev}: SWMM completed")
    
    COST.swmm_time += time.perf_counter() - t0_swmm
    COST.n_swmm_calls += len(EVENTS)  # Revised: count all 4 events
    
    # Measure PCA processing time
    t0_pca = time.perf_counter()
    for event in EVENTS:
        depth_data = results[event]
        model_pcs = {}
        
        for node in NODES:
            df_depth = depth_data[node]
            x_feat = preprocess_depth_series(df_depth)
            
            scaler, pca = pca_models[node]
            x_scaled = scaler.transform(x_feat.reshape(1, -1))
            x_pc = pca.transform(x_scaled)[0]
            
            n_pc = len(x_pc)
            for i in range(n_pc):
                model_pcs[f"{node}_PC{i+1}"] = float(x_pc[i])
        
        obs_pc = obs_pcs[event]
        J_e = compute_weighted_rmse(model_pcs, obs_pc, evr_weights)
        J_events[event] = J_e
        
        log(f"    ├─ {event}: J_e = {J_e:.6f}")
    
    COST.pca_time += time.perf_counter() - t0_pca
    
    J_total = float(np.mean(list(J_events.values())))
    log(f"    └─ J_total = {J_total:.6f}")
    
    return J_total, J_events


# ===== GP-BO =====
def make_gp_kernel(d: int):
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d), nu=2.5) \
           + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-3))

def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


# ===== Main =====
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ensure_dir(RESULT_ROOT / ts)
    best_dir = ensure_dir(result_dir / "best_theta")
    
    clean_temp()
    COST.start()
    
    log("=" * 80)
    log("Case B: LHS-assisted GP-BO with Direct SWMM Execution")
    log(f"N_INIT={N_INIT}, N_ITER={N_ITER}, PATIENCE={PATIENCE}")
    log("=" * 80)
    
    log("\n[LOAD] Loading data...")
    pca_models = load_pca_models()
    obs_pcs = load_obs_pcs()
    evr_weights = load_evr_weights()
    
    # Reuse LHS
    history = []
    COST.mark_init_start()
    X, y, hist = load_random_lhs_init(pca_models, obs_pcs, evr_weights, seed=42)
    history.append(hist)
    _g_history.append(hist)
    COST.mark_init_end()
    J_best = float(y[0])
    theta_best = X[0].copy()
    _g_theta_best = theta_best
    _g_J_best = J_best
    COST.mark_best_found(1)
    log(f"  θ_best = {', '.join([f'{FEATURES[i]}={theta_best[i]:.4f}' for i in range(len(FEATURES))])}")
    log(f"\n[BO] Start | J_best={J_best:.6f}")
    
    # BO
    log(f"\n[BO] Iteration start (max {N_ITER}, patience={PATIENCE})...")
    
    no_improve_count = 0
    
    for it in range(1, N_ITER + 1):
        iter_start_time = time.perf_counter()
        log(f"\n[BO] Iter {it}/{N_ITER}")
        
        # GP training (cost tracking)
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
        
        # EI maximization (cost tracking)
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
        
        try:
            J_next, J_events = evaluate_theta(theta_dict, f"bo{it:03d}", pca_models, obs_pcs, evr_weights)
        except Exception as e:
            log(f"  [ERROR] Evaluation failed: {e}")
            J_next = 999.0
            J_events = {ev: 999.0 for ev in EVENTS}
        
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
            COST.mark_best_found(1 + it)
            
            log(f"  ⬇ NEW BEST! (improvement: {improvement:.2f}%)")
            
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
            "iter": 1 + it,
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
            save_checkpoint(f"autosave_iter_{1 + it}")
        
        log(f"  Current J_best = {J_best:.6f} | no improvement: {no_improve_count}/{PATIENCE}")
        
        # Early Stopping
        if no_improve_count >= PATIENCE:
            log(f"\n[STOP] Early stop: no improvement for the last {PATIENCE} iterations")
            COST.converged_iter = N_INIT + it
            break
    
    if COST.converged_iter is None:
        COST.converged_iter = N_INIT + N_ITER
    
    # Save results
    log("\n[SAVE] Saving results...")
    
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(result_dir / "history.csv", index=False)
    
    best_payload = {
        "theta_star": {feat: float(theta_best[i]) for i, feat in enumerate(FEATURES)},
        "J_star": float(J_best),
        "n_init": N_INIT,
        "n_iter_total": len(history) - N_INIT,
        "final_iter": len(history),
        "bounds": {feat: {"lo": BOUNDS[i][0], "hi": BOUNDS[i][1]} for i, feat in enumerate(FEATURES)},
    }
    with open(result_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)
    
    # Save cost
    cost_data = COST.to_dict()
    with open(result_dir / "cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2, ensure_ascii=False)
    
    # Convergence plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        iter_nums = df_hist["iter"].values
        J_totals = df_hist["J_total"].values
        J_best_curve = np.minimum.accumulate(J_totals)
        
        plt.plot(iter_nums, J_totals, 'o-', alpha=0.5, label='J_total')
        plt.plot(iter_nums, J_best_curve, 'r-', linewidth=2, label='J_best')
        plt.axvline(N_INIT, color='gray', linestyle='--', label='INIT → BO')
        
        plt.xlabel('Iteration')
        plt.ylabel('J_total (EVR-weighted RMSE)')
        plt.title('CASE 55: GP-BO Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(result_dir / "convergence.png", dpi=150)
    except:
        pass
    
    clean_temp()
    
    log("\n" + "=" * 80)
    log("Case B optimization completed!")
    log("=" * 80)
    log(f"Best parameters:")
    for feat, val in zip(FEATURES, theta_best):
        log(f"  - {feat}: {val:.6f}")
    log(f"Best objective: J* = {J_best:.6f}")
    log("\nCost summary:")
    log(f"  Total elapsed: {cost_data['total_time_sec']:.2f} sec ({cost_data['total_time_sec']/60:.1f} min)")
    log(f"  SWMM time: {cost_data['swmm_time_sec']:.2f} sec ({cost_data['swmm_time_sec']/60:.1f} min)")
    log(f"  GP fit time: {cost_data['gp_fit_time_sec']:.2f} sec")
    log(f"  EI opt time: {cost_data['ei_opt_time_sec']:.2f} sec")
    log(f"  PCA time: {cost_data['pca_time_sec']:.2f} sec")
    log(f"  SWMM calls: {cost_data['n_swmm_calls']} calls")
    log(f"  Avg SWMM time: {cost_data['avg_swmm_per_call_sec']:.3f} sec/call")
    log(f"  GP fits: {cost_data['n_gp_fits']} fits")
    log(f"  Converged iteration: {cost_data['converged_iter']}")
    log(f"\nResult folder: {result_dir}")


if __name__ == "__main__":
    main()