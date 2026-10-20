"""Owner: Person 4. Re-run every runner config and compare the numbers with results/results.csv.

Usage (repo root, data/processed in place):
    python src/eval/reproduce.py                 # all configs
    python src/eval/reproduce.py --only rotate popularity
Writes nothing. Exit code 0 only if every config reproduces within --tol.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ["popularity", "rotate", "rgcn_distmult", "distmult_nograph", "vrgae"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="config names without .yaml")
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()
    names = args.only or CONFIGS + (["vrgae_checkpoint"] if (ROOT / "results" / "vrgae_checkpoint.pt").exists() else [])

    summary = []
    for n in names:
        cfg = f"configs/{n}.yaml"
        print(f"\n##### {cfg}", flush=True)
        rc = subprocess.run([sys.executable, "src/eval/run_experiment.py", "--config", cfg, "--check",
                             "--tol", str(args.tol)], cwd=ROOT).returncode
        summary.append((n, {0: "PASS", 1: "FAIL", 2: "NO LOGGED RUN"}.get(rc, f"ERROR({rc})")))
    print("\n===== reproducibility summary =====")
    for n, s in summary:
        print(f"{n:20s} {s}")
    sys.exit(0 if all(s == "PASS" for _, s in summary) else 1)


if __name__ == "__main__":
    main()
