# apply_and_run_caseA.py
from pathlib import Path
import re
from pyswmm import Simulation

# ===== Path settings =====
BASE = Path(r"C:\Users\SGJEONG99\Desktop\new\COMPARE\1_")
INP_DIR = BASE / "SUMMARY" / "SWMM"
OUT_DIR = BASE / "SUMMARY" / "SWMM_RESULTS"

EVENTS = ["R1", "R2", "R5", "R6", "R8"]

# CASE A parameters
THETA = {
    "Imperv": 1.020130854,
    "Width": 1.185595882,
    "Nimp": 0.960574979,
    "n_pipe": 0.725663439
}

# Deep-tunnel exclusion list
EXCLUDE_IDS = {
    "L335", "L454", "L456.1", "L456", "L454.1",
    "L453", "L453.2", "L453.1", "L453.1.2",
    "L477", "L326", "MH4306#1"
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Utilities =====
def read_text(p):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_text(p, t):
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)

def split_ws(line):
    return re.split(r'(\s+)', line.rstrip("\n"))

def join_ws(parts):
    return "".join(parts)

def token_at(parts, idx):
    pos = 2 * idx
    return (parts[pos], pos) if pos < len(parts) else (None, None)

def find_section(lines, name):
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

def data_idxs(sec):
    out = []
    for i in range(1, len(sec)):
        s = sec[i].strip()
        if s and not s.startswith(";"):
            out.append(i)
    return out

def as_float(x):
    try:
        return float(x)
    except:
        return None

def format_val(orig, val):
    m = re.match(r'^-?\d+(?:\.(\d+))?$', (orig or "").strip())
    if m:
        dec = len(m.group(1)) if m.group(1) else 0
        return f"{val:.{dec}f}"
    return f"{val:.6g}"

def should_skip(name):
    return (name or "").strip().upper() in {x.upper() for x in EXCLUDE_IDS}

# ===== INP modification =====
def apply_theta_inplace(inp_path, theta):
    text = read_text(inp_path)
    lines = text.splitlines()

    imp_sc = float(theta["Imperv"])
    wid_sc = float(theta["Width"])
    nimp_sc = float(theta["Nimp"])
    npipe_sc = float(theta["n_pipe"])

    # SUBCATCHMENTS
    s_subc = find_section(lines, "SUBCATCHMENTS")
    if None not in s_subc:
        sec = lines[s_subc[0]:s_subc[1]]
        new_sec = sec[:]
        for i in data_idxs(sec):
            parts = split_ws(new_sec[i])
            name, _ = token_at(parts, 0)
            if should_skip(name):
                continue
            
            t_imp, pos_imp = token_at(parts, 4)
            base_imp = as_float(t_imp)
            if base_imp is not None and pos_imp is not None:
                new_imp = max(0.0, min(100.0, base_imp * imp_sc))
                parts[pos_imp] = format_val(t_imp, new_imp)
            
            t_wid, pos_wid = token_at(parts, 5)
            base_wid = as_float(t_wid)
            if base_wid is not None and pos_wid is not None:
                parts[pos_wid] = format_val(t_wid, base_wid * wid_sc)
            
            new_sec[i] = join_ws(parts)
        lines[s_subc[0]:s_subc[1]] = new_sec

    # SUBAREAS
    s_suba = find_section(lines, "SUBAREAS")
    if None not in s_suba:
        sec = lines[s_suba[0]:s_suba[1]]
        new_sec = sec[:]
        for i in data_idxs(sec):
            parts = split_ws(new_sec[i])
            name, _ = token_at(parts, 0)
            if should_skip(name):
                continue
            
            t_nimp, pos_nimp = token_at(parts, 1)
            base_nimp = as_float(t_nimp)
            if base_nimp is not None and pos_nimp is not None:
                parts[pos_nimp] = format_val(t_nimp, base_nimp * nimp_sc)
            
            new_sec[i] = join_ws(parts)
        lines[s_suba[0]:s_suba[1]] = new_sec

    # CONDUITS
    s_cond = find_section(lines, "CONDUITS")
    if None not in s_cond:
        sec = lines[s_cond[0]:s_cond[1]]
        new_sec = sec[:]
        for i in data_idxs(sec):
            parts = split_ws(new_sec[i])
            name, _ = token_at(parts, 0)
            if should_skip(name):
                continue
            
            t_rgh, pos_rgh = token_at(parts, 4)
            base_rgh = as_float(t_rgh)
            if base_rgh is not None and pos_rgh is not None:
                parts[pos_rgh] = format_val(t_rgh, base_rgh * npipe_sc)
            
            new_sec[i] = join_ws(parts)
        lines[s_cond[0]:s_cond[1]] = new_sec

    write_text(inp_path, "\n".join(lines) + "\n")

# ===== Run SWMM =====
def run_swmm(inp_path, ev):
    out_path = OUT_DIR / f"10mm_{ev}.out"
    rpt_path = OUT_DIR / f"10mm_{ev}.rpt"
    
    try:
        with Simulation(str(inp_path), reportfile=str(rpt_path), outputfile=str(out_path)) as sim:
            for _ in sim:
                pass
        return True, ""
    except Exception as e:
        return False, str(e)

# ===== Main =====
def main():
    print(f"[INFO] CASE A Theta:")
    for k, v in THETA.items():
        print(f"  {k}: {v}")
    print()

    for ev in EVENTS:
        inp_path = INP_DIR / f"10mm_{ev}.inp"
        if not inp_path.exists():
            print(f"[SKIP] {ev}: INP file not found")
            continue
        
        print(f"[{ev}] Applying theta...")
        try:
            apply_theta_inplace(inp_path, THETA)
            print(f"[{ev}] Theta applied successfully")
        except Exception as e:
            print(f"[ERROR] {ev}: {e}")
            continue
        
        print(f"[{ev}] Running SWMM...")
        ok, msg = run_swmm(inp_path, ev)
        if ok:
            print(f"[{ev}] Completed\n")
        else:
            print(f"[ERROR] {ev}: {msg}\n")

    print(f"[DONE] CASE A finished")
    print(f"  INP: {INP_DIR}")
    print(f"  OUT: {OUT_DIR}")

if __name__ == "__main__":
    main()
