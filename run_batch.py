"""Batch command runner: reads commands from a file and executes them sequentially.

Usage:
    python run_batch.py commands.txt
    python run_batch.py commands.txt --stop-on-error
    python run_batch.py commands.txt --output results/

File format: one shell command per line, blank lines and # comments are ignored.

Example commands.txt:
    # Scan different ion counts
    python -m collision_pressure simulate --n-ions 10 --n-simulations 50 --trap-freq 0.5 1.0 0.1
    python -m collision_pressure simulate --n-ions 20 --n-simulations 50 --trap-freq 0.5 1.0 0.1
    python -m collision_pressure simulate --n-ions 30 --n-simulations 50 --trap-freq 0.5 1.0 0.1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_batch(path: Path, stop_on_error: bool = False, dry_run: bool = False) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    commands = []
    for raw in lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            commands.append(stripped)

    if not commands:
        print("No commands found in file.")
        return 0

    print(f"Loaded {len(commands)} commands from {path}")
    if dry_run:
        for i, cmd in enumerate(commands, 1):
            print(f"  [{i}] {cmd}")
        print("\n(dry run, not executing)")
        return 0

    n_ok = 0
    n_fail = 0
    t_total = time.time()

    for i, cmd in enumerate(commands, 1):
        tag = f"[{i}/{len(commands)}]"
        print(f"\n{'='*60}")
        print(f"{tag} {cmd}")
        print(f"{'='*60}")

        t0 = time.time()
        result = subprocess.run(cmd, shell=True)
        elapsed = time.time() - t0

        if result.returncode == 0:
            n_ok += 1
            status = "OK"
        else:
            n_fail += 1
            status = f"FAILED (exit {result.returncode})"
            if stop_on_error:
                print(f"\n{tag} {status} ({elapsed:.1f}s) — stopping")
                break

        print(f"\n{tag} {status} ({elapsed:.1f}s)")

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done: {n_ok} ok, {n_fail} failed, {total:.1f}s total")
    print(f"{'='*60}")

    return 1 if n_fail > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of shell commands from a text file",
    )
    parser.add_argument("file", type=Path, help="Text file with one command per line")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop on first failure (default: continue)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        return 1

    return run_batch(args.file, stop_on_error=args.stop_on_error, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
