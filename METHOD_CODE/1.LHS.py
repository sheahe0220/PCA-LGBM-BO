# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import csv
import random
from pathlib import Path
import re
from datetime import datetime, timedelta


# ===== User configuration =====
BASE_DIR   = Path(r"C:\Users\SGJEONG99\Desktop\new")
PARAM_CSV  = BASE_DIR / "input" / "para_1.csv"
TEMPLATE   = BASE_DIR / "swmm" / "10mm_R.inp"
OUT_ROOT   = BASE_DIR / "LHS_R"

DAT_DIR    = BASE_DIR / "input" / "DAT"   # R1.DAT, R2.DAT, ...
EVENTS     = ["R1", "R2", "R5", "R6", "R8"]
N_SAMPLES  = 300
RNG_SEED   = 42  # Seed for LHS reproducibility


# ===== Deep tunnel–related objects: protected from modification =====
SKIP_IDS = {s.upper() for s in [
    "L335", "L454", "L456.1", "L456", "L454.1",
    "L453", "L453.2", "L453.1", "L453.1.2",
    "L477", "L326", "MH4306#1"
]}


def should_skip(name: str | None) -> bool:
    if not name:
        return False
    return name.strip().upper() in SKIP_IDS


# ===== Common utilities =====
def read_text(p: Path) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_text(p: Path, t: str) -> None:
    with open(p, "w", encoding="utf-8") as f:
        f.write(t)


def find_section(lines: list[str], name: str):
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


def data_line_indexes(sec_lines: list[str]) -> list[int]:
    out = []
    for i in range(1, len(sec_lines)):
        s = sec_lines[i].strip()
        if (not s) or s.startswith(";"):
            continue
        out.append(i)
    return out


def split_keep_ws(line: str):
    return re.split(r'(\s+)', line.rstrip("\n"))


def join_keep_ws(parts) -> str:
    return "".join(parts)


def token_at(parts, token_idx: int):
    pos = 2 * token_idx
    if pos >= len(parts):
        return None, None
    return parts[pos], pos


# ===== Latin Hypercube Sampling =====
def lhs(n_samples: int, n_dim: int, seed: int = 42) -> list[list[float]]:
    random.seed(seed)
    result = [[0.0] * n_dim for _ in range(n_samples)]

    for j in range(n_dim):
        cut_points = [k / n_samples for k in range(n_samples + 1)]
        intervals = [(cut_points[k], cut_points[k + 1]) for k in range(n_samples)]
        u = [random.uniform(a, b) for (a, b) in intervals]
        random.shuffle(u)
        for i in range(n_samples):
            result[i][j] = u[i]

    return result


# ===== Read para_1.csv =====
def load_params(csv_path: Path):
    meta = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            low = float(row["low"])
            high = float(row["high"])
            ptype = row.get("type", "abs").strip().lower()
            meta.append({"name": name, "low": low, "high": high, "type": ptype})
    return meta


def generate_lhs_samples(meta, n_samples: int, seed: int = 42):
    d = len(meta)
    u_mat = lhs(n_samples, d, seed=seed)
    header = [p["name"] for p in meta]
    samples = []

    for i in range(n_samples):
        row = {}
        for j, p in enumerate(meta):
            u = u_mat[i][j]
            low = p["low"]
            high = p["high"]
            val = low + u * (high - low)
            row[p["name"]] = val
        samples.append(row)
    return samples, header


# ===== Modify [TIMESERIES] / [RAINGAGES] =====
def build_timeseries_block_for_event(base_lines: list[str], ev: str) -> list[str]:
    s, e = find_section(base_lines, "TIMESERIES")
    base_path = None

    if s is not None:
        sec = base_lines[s:e]
        idxs = data_line_indexes(sec)
        for i in idxs:
            parts = split_keep_ws(sec[i])
            name, _ = token_at(parts, 0)
            if not name:
                continue
            if name.upper().startswith("R"):
                file_tok, _ = token_at(parts, 2)
                if file_tok and file_tok.startswith('"'):
                    base_path = file_tok.strip('"')
                    break

    if base_path is None:
        dat_path = str((DAT_DIR / f"{ev}.DAT").resolve())
    else:
        base_dir = os.path.dirname(base_path)
        dat_path = os.path.join(base_dir, f"{ev}.DAT")

    return [
        "[TIMESERIES]",
        ";;Name           Date       Time       Value",
        ";;-------------- ---------- ---------- ----------",
        f'{ev:<16}FILE "{dat_path}"',
        ""
    ]


def apply_timeseries_and_raingage_for_event(lines: list[str], ev: str) -> None:
    # 1) TIMESERIES replacement
    s_ts, e_ts = find_section(lines, "TIMESERIES")
    ts_block = build_timeseries_block_for_event(lines, ev)
    if s_ts is None:
        lines.extend(ts_block)
    else:
        lines[s_ts:e_ts] = ts_block

    # 2) RAINGAGES update or creation
    s_rg, e_rg = find_section(lines, "RAINGAGES")
    line_template = f'rain             VOLUME    0:10     1.0      TIMESERIES {ev}'
    if s_rg is None:
        lines += [
            "[RAINGAGES]",
            ";;Name           Format    Interval SCF      Source",
            line_template,
            ""
        ]
        return

    sec = lines[s_rg:e_rg]
    idxs = data_line_indexes(sec)
    for i in idxs:
        parts = split_keep_ws(sec[i])
        name, _ = token_at(parts, 0)
        if name == "rain":
            fmt, _   = token_at(parts, 1)
            inter, _ = token_at(parts, 2)
            scf, _   = token_at(parts, 3)
            sec[i] = f"{name:<16}{(fmt or 'VOLUME'):<10}{(inter or '0:10'):<8}{(scf or '1.0'):<8}TIMESERIES {ev}"
            lines[s_rg:e_rg] = sec
            return

    sec.append(line_template)
    lines[s_rg:e_rg] = sec


# ===== Read event start/end time from DAT =====
def parse_datetime_from_tokens(date_str: str, time_str: str) -> datetime:
    formats = [
        ("%Y-%m-%d", "%H:%M:%S"), ("%Y-%m-%d", "%H:%M"),
        ("%m/%d/%Y", "%H:%M:%S"), ("%m/%d/%Y", "%H:%M"),
        ("%Y/%m/%d", "%H:%M:%S"), ("%Y/%m/%d", "%H:%M"),
        ("%m/%d/%y", "%H:%M:%S"), ("%m/%d/%y", "%H:%M"),
    ]
    for df, tf in formats:
        try:
            return datetime.strptime(f"{date_str} {time_str}", f"{df} {tf}")
        except ValueError:
            continue
    raise ValueError(f"Unsupported DAT date/time format: {date_str} {time_str}")


def get_event_start_end_from_dat(ev: str):
    dat_path = DAT_DIR / f"{ev}.DAT"
    if not dat_path.exists():
        raise FileNotFoundError(f"DAT file not found for {ev}: {dat_path}")

    start_dt, end_dt = None, None
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                dt = parse_datetime_from_tokens(parts[0], parts[1])
            except ValueError:
                continue
            start_dt = dt if start_dt is None else start_dt
            end_dt = dt

    if start_dt is None or end_dt is None:
        raise RuntimeError(f"No valid timestamps found in {dat_path}")

    end_dt += timedelta(hours=6)

    return (
        start_dt.strftime("%m/%d/%Y"),
        start_dt.strftime("%H:%M:%S"),
        end_dt.strftime("%m/%d/%Y"),
        end_dt.strftime("%H:%M:%S"),
    )


def apply_options_for_event(lines: list[str], ev: str) -> None:
    try:
        s_date, s_time, e_date, e_time = get_event_start_end_from_dat(ev)
    except Exception as e:
        print(f"[WARN] Failed to update OPTIONS for {ev} (keeping existing values): {e}")
        return

    s, e = find_section(lines, "OPTIONS")
    if s is None:
        return

    sec = lines[s:e]
    idxs = data_line_indexes(sec)
    for i in idxs:
        parts = split_keep_ws(sec[i])
        key, _ = token_at(parts, 0)
        if key in ("START_DATE", "REPORT_START_DATE"):
            _, pos = token_at(parts, 1)
            if pos is not None:
                parts[pos] = s_date
        elif key == "END_DATE":
            _, pos = token_at(parts, 1)
            if pos is not None:
                parts[pos] = e_date
        elif key in ("START_TIME", "REPORT_START_TIME"):
            _, pos = token_at(parts, 1)
            if pos is not None:
                parts[pos] = s_time
        elif key == "END_TIME":
            _, pos = token_at(parts, 1)
            if pos is not None:
                parts[pos] = e_time
        sec[i] = join_keep_ws(parts)

    lines[s:e] = sec


# ===== Main =====
def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template INP not found: {TEMPLATE}")
    if not PARAM_CSV.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {PARAM_CSV}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta = load_params(PARAM_CSV)
    samples, header = generate_lhs_samples(meta, N_SAMPLES, seed=RNG_SEED)

    lhs_csv = OUT_ROOT / "lhs_samples_10d.csv"
    with lhs_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"] + header)
        for i, row in enumerate(samples, start=1):
            writer.writerow([f"sample_{i:03d}"] + [row[h] for h in header])
    print(f"[INFO] LHS sample CSV written: {lhs_csv}")

    base_lines = read_text(TEMPLATE).splitlines()

    event_templates = {}
    for ev in EVENTS:
        lines_ev = list(base_lines)
        apply_timeseries_and_raingage_for_event(lines_ev, ev)
        apply_options_for_event(lines_ev, ev)
        event_templates[ev] = lines_ev
        print(f"[INFO] Event template prepared: {ev}")

    for ev in EVENTS:
        ev_dir = OUT_ROOT / ev
        ev_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[EVENT] {ev} → INP generation started (N={N_SAMPLES})")

        for i, p in enumerate(samples, start=1):
            lines = list(event_templates[ev])
            out_inp = ev_dir / f"sample_{i:03d}.inp"
            write_text(out_inp, "\n".join(lines) + "\n")
            if i % 50 == 0 or i == N_SAMPLES:
                print(f"  - {ev}: sample_{i:03d}.inp completed")

    print("\n[DONE] All LHS-based INP files and audit outputs have been generated.")
    print(f"Output root directory: {OUT_ROOT}")


if __name__ == "__main__":
    main()
