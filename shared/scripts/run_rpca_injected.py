#!/usr/bin/env python3
"""Run RPCA on all injected_meters and save canonical scores to runs/rpca/injected."""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INJECTED_DIR = PROJECT_ROOT / "data" / "injected_meters"
OUT_DIR = PROJECT_ROOT / "runs" / "rpca" / "injected"
RPCA_SCRIPT = PROJECT_ROOT / "methods" / "rpca" / "src" / "rpca_spartan_train_test.py"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def process_file(csv_path: Path):
    cmd = [
        sys.executable,
        str(RPCA_SCRIPT),
        "--csv", str(csv_path),
        "--outdir", str(OUT_DIR),
        "--test-start", "2026-08-01",
        "--test-end", "2026-09-01",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error on {csv_path.name}: {res.stderr}")
        return False
    return True

def main():
    csv_files = sorted(p for p in INJECTED_DIR.glob("*.csv") if not p.name.startswith("_"))
    print(f"Running RPCA on {len(csv_files)} files using multiprocessing...")
    with Pool(processes=4) as pool:
        results = pool.map(process_file, csv_files)
    success = sum(results)
    print(f"RPCA completed: {success}/{len(csv_files)} succeeded. Outputs saved in {OUT_DIR}")

if __name__ == "__main__":
    main()
