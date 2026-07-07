# -*- coding: utf-8 -*-

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pyswmm import Simulation

# ===== Basic configuration =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new")

LHS_INP_ROOT = BASE / "LHS_R"
RES_ROOT     = BASE / "LHS_RESULTS"

# Calibration events only (R8 excluded for validation)
EVENTS = ["R1", "R2", "R5", "R6"]

# Parallel execution settings
USE_PARALLEL = True
N_WORKERS    = 6   # Set 0 or 1 for sequential execution


def run_single_sim(event: str, inp_path: Path) -> tuple[str, str, bool, str]:
    run_id = inp_path.stem  # e.g., sample_001
    out_dir = RES_ROOT / event
    out_dir.mkdir(parents=True, exist_ok=True)

    rpt_path = out_dir / f"{run_id}.rpt"
    out_path = out_dir / f"{run_id}.out"

    try:
        print(f"[RUN] {event} | {run_id}")
        with Simulation(
            str(inp_path),
            reportfile=str(rpt_path),
            outputfile=str(out_path)
        ) as sim:
            for _ in sim:
                pass
        return event, run_id, True, ""
    except Exception as e:
        return event, run_id, False, str(e)


def main():
    RES_ROOT.mkdir(parents=True, exist_ok=True)

    # Build job list: all (event × INP) combinations
    jobs: list[tuple[str, Path]] = []
    for ev in EVENTS:
        inp_dir = LHS_INP_ROOT / ev
        if not inp_dir.exists():
            print(f"[WARN] INP directory not found: {inp_dir} (skipped)")
            continue

        inps = sorted(inp_dir.glob("sample_*.inp"))
        if not inps:
            print(f"[WARN] {ev}: no sample_*.inp found (skipped)")
            continue

        print(f"[INFO] {ev}: {len(inps)} INP files found ({inp_dir})")
        for inp in inps:
            jobs.append((ev, inp))

    print(f"[INFO] Total simulation jobs: {len(jobs)}")

    if not jobs:
        print("[ERROR] No INP files to run. Check the paths.")
        return

    # ===== Sequential execution =====
    if not USE_PARALLEL or N_WORKERS <= 1:
        print("[INFO] Sequential execution mode")
        for ev, inp in jobs:
            event, run_id, ok, msg = run_single_sim(ev, inp)
            if not ok:
                print(f"[ERROR] {event} | {run_id}: {msg}")
        print("[DONE] All simulations completed (sequential mode).")
        return

    # ===== Parallel execution =====
    print(f"[INFO] Parallel execution started (workers={N_WORKERS})")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as exe:
        future_to_job = {
            exe.submit(run_single_sim, ev, inp): (ev, inp)
            for ev, inp in jobs
        }

        n_ok, n_err = 0, 0
        for fut in as_completed(future_to_job):
            ev, inp = future_to_job[fut]
            try:
                event, run_id, ok, msg = fut.result()
            except Exception as e:
                print(f"[ERROR] {ev} | {inp.stem}: exception occurred - {e}")
                n_err += 1
                continue

            if ok:
                n_ok += 1
            else:
                n_err += 1
                print(f"[ERROR] {event} | {run_id}: {msg}")

        print(f"[DONE] Parallel execution finished - success: {n_ok}, failed: {n_err}")


if __name__ == "__main__":
    main()
