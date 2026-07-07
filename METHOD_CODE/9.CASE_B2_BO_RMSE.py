# -*- coding: utf-8 -*-
# 9.CASE_B2_BO_RMSE.py
# ------------------------------------------------------------
# Case B-2: GP-BO with random init + direct time-domain RMSE objective
# Purpose: isolate the objective-function effect (vs Case B-1 EVR)
# Same init point, same BO settings, only the objective differs
# Uses min-baseline OBS
# ------------------------------------------------------------
from __future__ import annotations
import os, time, json, re, shutil, signal
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyswmm import Simulation
from pyswmm.output import Output
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore")

# ===== USER CONFIG: change paths here =====
BASE         = Path(r"C:\Users\SGJEONG99\Desktop\new")
TEMPLATE_DIR = BASE / "SWMM"            # event templates: 10mm_{event}.inp

# Min-baseline OBS (per event-node files)
OBS_MINBASE_DIR = BASE / "OBS_min_baseline"

RESULT_ROOT  = BASE / "RESULTS" / "CaseB2_RMSE"
TEMP_DIR     = RESULT_ROOT / "temp"

SENSOR_NODE = {"15-0003": "MH0126", "15-0005": "M0113"}
EVENTS = ["R1", "R2", "R5", "R6"]
NODES  = ["M0113", "MH0126"]

EXCLUDE_SUBCATCHMENTS = {"MH4306#1"}
EXCLUDE_CONDUITS = {
    "L335", "L454", "L456.1", "L456", "L454.1",
    "L453", "L453.2", "L453.1", "L453.1.2", "L477", "L326"
}

# ===== BO settings (IDENTICAL to Case B-1) =====
FEATURES = ["Imperv_scale", "Width_scale", "Nimp_scale", "n_pipe_scale"]
BOUNDS = [
    (0.7, 1.2),   # Imperv_scale
    (0.5, 1.5),   # Width_scale
    (0.8, 1.2),   # Nimp_scale
    (0.7, 1.3),   # n_pipe_scale
]

N_INIT           = 0
N_ITER           = 200
PATIENCE         = 40
MIN_IMPROVEMENT  = 0.001
N_WORKERS        = 4
RNG              = np.random.RandomState(42)
AUTOSAVE_EVERY   = 10


# ===== Cost Tracker =====
class CostTracker:
    def __init__(self):
        self.total_start = None
        self.swmm_time = 0.0
        self.gp_fit_time = 0.0
        self.ei_opt_time = 0.0
        self.n_swmm_calls = 0
        self.n_gp_fits = 0
        self.n_ei_opts = 0
        self.converged_iter = None
        self.init_start = None
        self.init_end = None
        self.best_found_at = None
        self.best_found_time = None
        self.best_swmm_calls_snapshot = None
        self.best_swmm_time_snapshot = None

    def start(self):         self.total_start = time.perf_counter()
    def elapsed(self):       return time.perf_counter() - self.total_start if self.total_start else 0.0
    def mark_init_start(self): self.init_start = time.perf_counter()
    def mark_init_end(self):   self.init_end = time.perf_counter()

    def mark_best_found(self, iter_num):
        if self.best_found_at is None or iter_num > self.best_found_at:
            self.best_found_at = iter_num
            self.best_found_time = self.elapsed()
            self.best_swmm_calls_snapshot = self.n_swmm_calls
            self.best_swmm_time_snapshot = self.swmm_time

    def get_init_time(self):
        return (self.init_end - self.init_start) if (self.init_start and self.init_end) else 0.0

    def to_dict(self):
        return {
            "total_time_sec": self.elapsed(),
            "init_time_sec": self.get_init_time(),
            "best_found_time_sec": self.best_found_time or 0.0,
            "best_swmm_calls_at_best": self.best_swmm_calls_snapshot,
            "best_swmm_time_at_best": self.best_swmm_time_snapshot,
            "swmm_time_sec": self.swmm_time,
            "gp_fit_time_sec": self.gp_fit_time,
            "ei_opt_time_sec": self.ei_opt_time,
            "n_swmm_calls": self.n_swmm_calls,
            "n_gp_fits": self.n_gp_fits,
            "n_ei_opts": self.n_ei_opts,
            "converged_iter": self.converged_iter,
            "best_found_at_iter": self.best_found_at,
            "n_init": N_INIT, "n_iter_max": N_ITER,
            "avg_swmm_per_call_sec": self.swmm_time / max(1, self.n_swmm_calls),
            "avg_gp_fit_sec": self.gp_fit_time / max(1, self.n_gp_fits),
            "avg_ei_opt_sec": self.ei_opt_time / max(1, self.n_ei_opts),
            "mode": "CaseB2_random_direct_RMSE",
        }

COST = CostTracker()

# ===== Global state for checkpoint =====
_g_history = []
_g_result_dir = None
_g_best_dir = None
_g_theta_best = None
_g_J_best = None
_g_interrupted = False


# ===== Utilities =====
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def safe_remove(p: Path):
    try:
        if p.exists(): p.unlink()
    except: pass

def clean_temp():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        time.sleep(0.2)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(reason: str = "checkpoint"):
    if not _g_history or _g_result_dir is None:
        return
    log(f"[SAVE] {reason} ({len(_g_history)} iters)...")
    try:
        pd.DataFrame(_g_history).to_csv(_g_result_dir / "history.csv", index=False)
        cost_data = COST.to_dict()
        cost_data["save_reason"] = reason
        with open(_g_result_dir / "cost_summary.json", "w", encoding="utf-8") as f:
            json.dump(cost_data, f, indent=2, ensure_ascii=False)
        if _g_theta_best is not None:
            payload = {
                "theta_star": {feat: float(_g_theta_best[i]) for i, feat in enumerate(FEATURES)},
                "J_star": float(_g_J_best),
                "n_iter_completed": len(_g_history),
                "bounds": {feat: {"lo": BOUNDS[i][0], "hi": BOUNDS[i][1]} for i, feat in enumerate(FEATURES)},
                "interrupted": _g_interrupted,
                "mode": "CaseB2_random_direct_RMSE",
            }
            with open(_g_result_dir / "best_result.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        log(f"  -> saved")
    except Exception as e:
        log(f"  [WARN] save failed: {e}")

def signal_handler(signum, frame):
    global _g_interrupted
    log("\n[INTERRUPT] Ctrl+C - saving...")
    _g_interrupted = True
    save_checkpoint("interrupted")
    exit(0)


# ===== LHS =====
def latin_hypercube(n: int, d: int, rng) -> np.ndarray:
    U = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        U[:, j] = (perm + rng.rand(n)) / n
    return U

def scale_to_bounds(U: np.ndarray, bounds: list) -> np.ndarray:
    out = np.zeros_like(U)
    for i, (lo, hi) in enumerate(bounds):
        out[:, i] = lo + (hi - lo) * U[:, i]
    return out


# ===== INP handling =====
def read_inp(p: Path) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_inp(p: Path, text: str):
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)

def split_keep_ws(line: str):
    return re.split(r'(\s+)', line.rstrip("\n"))

def join_keep_ws(parts) -> str:
    return "".join(parts)

def token_at(parts, idx: int):
    pos = 2 * idx
    return (parts[pos], pos) if pos < len(parts) else (None, None)

def parse_section(lines, name: str):
    s = None
    for i, l in enumerate(lines):
        if l.strip().upper() == f"[{name.upper()}]":
            s = i; break
    if s is None: return None, None
    e = len(lines)
    for j in range(s + 1, len(lines)):
        if re.match(r"\s*\[.+\]\s*$", lines[j]):
            e = j; break
    return s, e

def data_idxs(sec_lines):
    return [i for i in range(1, len(sec_lines))
            if sec_lines[i].strip() and not sec_lines[i].strip().startswith(";")]

def apply_theta_to_inp(template_text: str, theta: dict) -> str:
    lines = template_text.splitlines()
    imp_sc   = theta.get("Imperv_scale", 1.0)
    wid_sc   = theta.get("Width_scale",  1.0)
    nimp_sc  = theta.get("Nimp_scale",   1.0)
    npipe_sc = theta.get("n_pipe_scale", 1.0)

    s, e = parse_section(lines, "SUBCATCHMENTS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name, _ = token_at(parts, 0)
            if (name or "").strip() in EXCLUDE_SUBCATCHMENTS: continue
            for idx, sc, clamp in [(4, imp_sc, True), (5, wid_sc, False)]:
                tok, pos = token_at(parts, idx)
                if tok and pos:
                    try:
                        v = float(tok) * sc
                        if clamp: v = max(0.0, min(v, 100.0))
                        parts[pos] = f"{v:.6g}"
                    except: pass
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec

    s, e = parse_section(lines, "SUBAREAS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name, _ = token_at(parts, 0)
            if (name or "").strip() in EXCLUDE_SUBCATCHMENTS: continue
            tok, pos = token_at(parts, 1)
            if tok and pos:
                try: parts[pos] = f"{float(tok)*nimp_sc:.6g}"
                except: pass
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec

    s, e = parse_section(lines, "CONDUITS")
    if None not in (s, e):
        sec = lines[s:e]
        for i in data_idxs(sec):
            parts = split_keep_ws(sec[i])
            name, _ = token_at(parts, 0)
            if (name or "").strip() in EXCLUDE_CONDUITS: continue
            tok, pos = token_at(parts, 4)
            if tok and pos:
                try: parts[pos] = f"{float(tok)*npipe_sc:.6g}"
                except: pass
            sec[i] = join_keep_ws(parts)
        lines[s:e] = sec

    return "\n".join(lines)


# ===== SWMM execution =====
def run_swmm_event(event: str, theta: dict, iter_id: str):
    template_path = TEMPLATE_DIR / f"10mm_{event}.inp"
    if not template_path.exists():
        raise FileNotFoundError(f"Template INP not found: {template_path}")

    modified_text = apply_theta_to_inp(read_inp(template_path), theta)

    inp_path = TEMP_DIR / f"{iter_id}_{event}.inp"
    out_path = TEMP_DIR / f"{iter_id}_{event}.out"
    rpt_path = TEMP_DIR / f"{iter_id}_{event}.rpt"
    write_inp(inp_path, modified_text)

    try:
        with Simulation(str(inp_path), reportfile=str(rpt_path), outputfile=str(out_path)) as sim:
            for _ in sim: pass
    except Exception as e:
        raise RuntimeError(f"SWMM failed ({event}): {e}")

    try:
        from pyswmm.output import NodeAttribute
    except:
        NodeAttribute = None

    depth_data = {}
    with Output(str(out_path)) as out:
        for node in NODES:
            data = out.node_series(node, NodeAttribute.DEPTH if NodeAttribute else "DEPTH")
            if isinstance(data, dict):
                s = pd.Series(data)
                s.index = pd.to_datetime(s.index)
                s = s.sort_index().astype(float)
                depth_data[node] = s.rename("depth").to_frame().assign(time=s.index)[["time","depth"]].reset_index(drop=True)
            else:
                df = pd.DataFrame(list(data), columns=["time","depth"])
                df["time"] = pd.to_datetime(df["time"])
                depth_data[node] = df.sort_values("time").reset_index(drop=True)

    for f in [inp_path, out_path, rpt_path]:
        safe_remove(f)

    return event, depth_data


# ===== Min-baseline OBS loading =====
def load_obs_minbase() -> dict:
    """Load min-baseline OBS per event-node from pre-computed CSV files."""
    obs = {}
    for event in EVENTS:
        obs[event] = {}
        for node in NODES:
            path = OBS_MINBASE_DIR / event / f"{node}_obs_depth_{event}_minbase.csv"
            if not path.exists():
                raise FileNotFoundError(f"Min-baseline OBS not found: {path}")
            df = pd.read_csv(path)
            df["Time"] = pd.to_datetime(df["Time"])
            obs[event][node] = df.set_index("Time")["Depth_adjusted"].sort_index()
    return obs


# ===== Objective: Direct RMSE (no baseline removal needed) =====
def compute_rmse_direct(sim_depth: np.ndarray, obs_depth: np.ndarray) -> float:
    n = min(len(sim_depth), len(obs_depth))
    if n == 0: return 999.0
    diff = sim_depth[:n] - obs_depth[:n]
    return float(np.sqrt(np.mean(diff**2)))


def evaluate_theta(theta_dict: dict, iter_id: str, obs_minbase: dict):
    log(f"  theta = {theta_dict}")
    J_events = {}

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        futures = {exe.submit(run_swmm_event, ev, theta_dict, iter_id): ev for ev in EVENTS}
        results = {}
        for fut in as_completed(futures):
            ev = futures[fut]
            ev_out, depth_data = fut.result()
            results[ev_out] = depth_data
            log(f"    +-- {ev_out}: SWMM done")
    COST.swmm_time += time.perf_counter() - t0
    COST.n_swmm_calls += len(EVENTS)

    for event in EVENTS:
        depth_data = results[event]
        rmse_nodes = []

        for node in NODES:
            df_sim = depth_data[node]
            sim_arr = df_sim["depth"].to_numpy(dtype=float)
            model_times = pd.DatetimeIndex(df_sim["time"])

            # Get min-baseline OBS for this event-node
            obs_s = obs_minbase[event][node]

            # Align OBS to model times
            obs_aligned = obs_s.reindex(model_times, method="nearest",
                                         tolerance=pd.Timedelta("2min")).ffill().bfill()
            obs_arr = obs_aligned.to_numpy(dtype=float)

            # SIM: interpolate NaN, clip negative (SWMM starts ~0, no baseline removal)
            sim_proc = pd.Series(sim_arr).interpolate(limit_direction="both").to_numpy()
            sim_proc = np.maximum(sim_proc, 0.0)

            # OBS already baseline-adjusted (min-baseline), no further processing
            obs_proc = obs_arr.copy()
            obs_proc = np.where(np.isnan(obs_proc), 0.0, obs_proc)

            rmse = compute_rmse_direct(sim_proc, obs_proc)
            rmse_nodes.append(rmse)
            log(f"    +-- {event}-{node}: RMSE={rmse:.6f} m")

        J_e = float(np.mean(rmse_nodes)) if rmse_nodes else 999.0
        J_events[event] = J_e
        log(f"    +-- {event}: J_e={J_e:.6f}")

    J_total = float(np.mean(list(J_events.values())))
    log(f"    +-- J_total={J_total:.6f}")
    return J_total, J_events


# ===== GP kernel / EI =====
def make_gp_kernel(d: int):
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(d), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-3)))

def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma <= 0.0] = 0.0
    return ei


# ===== MAIN =====
def main():
    global _g_result_dir, _g_best_dir, _g_theta_best, _g_J_best

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = ensure_dir(RESULT_ROOT / ts)
    best_dir   = ensure_dir(result_dir / "best_theta")
    _g_result_dir = result_dir
    _g_best_dir   = best_dir

    signal.signal(signal.SIGINT, signal_handler)
    clean_temp()
    COST.start()

    N_TOTAL_STEPS = 4

    log("=" * 80)
    log("Case B-2: Random Init + Direct RMSE objective")
    log("  OBS: min-baseline")
    log(f"N_ITER={N_ITER}, PATIENCE={PATIENCE}, SEED=42")
    log(f"Result dir: {result_dir}")
    log("=" * 80)

    # [1/4] Load data
    print(f"\n[1/{N_TOTAL_STEPS} | 25%] Loading min-baseline OBS...")
    obs_minbase = load_obs_minbase()
    log(f"  OBS loaded: {list(obs_minbase.keys())} x {NODES}")

    # [2/4] Random init (same seed as B-1)
    print(f"\n[2/{N_TOTAL_STEPS} | 50%] Random init (seed=42)...")
    COST.mark_init_start()

    x0 = scale_to_bounds(latin_hypercube(1, len(FEATURES), RNG), BOUNDS)[0]
    theta_dict = {f: float(v) for f, v in zip(FEATURES, x0)}
    log(f"  Init theta: {theta_dict}")

    t0 = time.perf_counter()
    J_total, J_events = evaluate_theta(theta_dict, "init000", obs_minbase)

    X = np.array([x0])
    y = np.array([J_total])
    J_best = J_total
    theta_best = x0.copy()

    history = []
    hist = {"iter": 1, "stage": "INIT_RAND",
            **{f"theta_{f}": theta_dict[f] for f in FEATURES},
            **{f"J_{ev}": J_events[ev] for ev in EVENTS},
            "J_total": J_total, "is_best": 1,
            "iter_time_sec": time.perf_counter() - t0,
            "elapsed_time_sec": COST.elapsed()}
    history.append(hist)
    _g_history.append(hist)

    COST.mark_init_end()
    _g_theta_best = theta_best
    _g_J_best = J_best
    COST.mark_best_found(1)
    log(f"  [INIT] J_best={J_best:.6f}")

    # [3/4] BO loop
    print(f"\n[3/{N_TOTAL_STEPS} | 75%] GP-BO start (max {N_ITER} iters)...")
    no_improve_count = 0

    try:
        for it in tqdm(range(1, N_ITER + 1), desc="[BO] Optimization"):
            iter_t0 = time.perf_counter()

            t0 = time.perf_counter()
            gp = GaussianProcessRegressor(
                kernel=make_gp_kernel(len(FEATURES)),
                alpha=1e-6, normalize_y=True,
                n_restarts_optimizer=3, random_state=RNG)
            gp.fit(X, y)
            COST.gp_fit_time += time.perf_counter() - t0
            COST.n_gp_fits += 1

            t0 = time.perf_counter()
            best_ei, best_x = -np.inf, None
            for _ in range(10):
                x0_cand = RNG.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
                def obj_ei(x):
                    mu, sigma = gp.predict(np.array(x).reshape(1,-1), return_std=True)
                    return -expected_improvement(mu, sigma, J_best, xi=0.01)[0]
                res = minimize(obj_ei, x0=x0_cand, bounds=BOUNDS, method="L-BFGS-B")
                if -res.fun > best_ei:
                    best_ei = -res.fun
                    best_x = res.x
            COST.ei_opt_time += time.perf_counter() - t0
            COST.n_ei_opts += 1

            x_next = best_x
            theta_dict = {f: float(v) for f, v in zip(FEATURES, x_next)}

            try:
                J_next, J_events = evaluate_theta(theta_dict, f"bo{it:03d}", obs_minbase)
            except Exception as ex:
                log(f"  [ERROR] {ex}")
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
                COST.mark_best_found(len(history) + 1)
                log(f"  >> NEW BEST! J={J_best:.6f} (improv: {improvement:.2f}%)")
                for ev in EVENTS:
                    tmpl = read_inp(TEMPLATE_DIR / f"10mm_{ev}.inp")
                    write_inp(best_dir / f"best_iter{it:03d}_{ev}.inp",
                              apply_theta_to_inp(tmpl, theta_dict))
            else:
                no_improve_count += 1

            iter_time = time.perf_counter() - iter_t0
            hist = {"iter": len(history)+1, "stage": "BO",
                    **{f"theta_{f}": theta_dict[f] for f in FEATURES},
                    **{f"J_{ev}": J_events[ev] for ev in EVENTS},
                    "J_total": J_next, "is_best": is_best,
                    "iter_time_sec": iter_time,
                    "elapsed_time_sec": COST.elapsed()}
            history.append(hist)
            _g_history.append(hist)

            if AUTOSAVE_EVERY > 0 and it % AUTOSAVE_EVERY == 0:
                save_checkpoint(f"autosave_iter_{it}")

            log(f"  J_best={J_best:.6f} | no_improve={no_improve_count}/{PATIENCE} | elapsed={COST.elapsed()/60:.1f}min")

            if no_improve_count >= PATIENCE:
                log(f"\n[STOP] Early stop: {PATIENCE} iters without improvement")
                COST.converged_iter = it
                break

        if COST.converged_iter is None:
            COST.converged_iter = N_ITER

    except KeyboardInterrupt:
        log("\n[INTERRUPT] Ctrl+C")
        save_checkpoint("interrupted")
        raise
    except Exception as ex:
        log(f"\n[ERROR] {ex}")
        save_checkpoint("error")
        raise
    finally:
        save_checkpoint("final")

    # [4/4] Save results
    print(f"\n[4/{N_TOTAL_STEPS} | 100%] Saving results...")
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(result_dir / "history.csv", index=False)

    best_payload = {
        "theta_star": {f: float(theta_best[i]) for i, f in enumerate(FEATURES)},
        "J_star": float(J_best),
        "n_iter_total": len(history),
        "bounds": {f: {"lo": BOUNDS[i][0], "hi": BOUNDS[i][1]} for i, f in enumerate(FEATURES)},
        "mode": "CaseB2_random_direct_RMSE",
    }
    with open(result_dir / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    cost_data = COST.to_dict()
    with open(result_dir / "cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(cost_data, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        iters = df_hist["iter"].values
        ax.plot(iters, df_hist["J_total"].values, "o-", alpha=0.4, label="J per iter")
        ax.plot(iters, np.minimum.accumulate(df_hist["J_total"].values), "r-", lw=2, label="J best")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE (m) - direct time domain")
        ax.set_title("Case B-2: Random Init + Direct RMSE Objective (min-baseline)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(result_dir / "convergence.png", dpi=150)
        plt.close(fig)
        log("  convergence.png saved")
    except Exception as ex:
        log(f"  [WARN] plot failed: {ex}")

    clean_temp()

    log("\n" + "=" * 80)
    log("Case B-2 DONE!")
    log("=" * 80)
    for f, v in zip(FEATURES, theta_best):
        log(f"  {f}: {v:.6f}")
    log(f"  J* = {J_best:.6f} m (RMSE)")
    log(f"  Total time: {cost_data['total_time_sec']/60:.1f} min")
    log(f"  SWMM calls: {cost_data['n_swmm_calls']}")
    log(f"  Converged at iter: {cost_data['converged_iter']}")
    log(f"  Result dir: {result_dir}")


if __name__ == "__main__":
    main()
