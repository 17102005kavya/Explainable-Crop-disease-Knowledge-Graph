"""Owner: Person 4. Small torch-free helpers for reproducibility: hashing, git commit, log lookup, comparison."""
import csv
import hashlib
import subprocess
from pathlib import Path

CHECK_COLS = ("MRR", "Hits@1", "Hits@3", "Hits@10", "AUC")


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(root) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _norm(p) -> str:
    return str(p).replace("\\", "/")


def latest_logged_rows(csv_path, config, seed) -> dict:
    """Rows ({relation: row}) of the newest runner run for this config and seed; {} if there is none.
    Legacy rows without an 'overall' entry (written by other scripts) are ignored."""
    runs = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if _norm(row.get("config", "")) != _norm(config) or str(row.get("seed")) != str(seed):
                continue
            runs.setdefault(row["run_id"], {})[row["relation"]] = row
    runs = {rid: rows for rid, rows in runs.items() if "overall" in rows}
    if not runs:
        return {}
    newest = max(runs, key=lambda rid: rid.rsplit("-", 1)[-1])      # run_id ends with a timestamp
    return runs[newest]


def compare_rows(new_rows, old_by_relation: dict, tol: float = 0.005):
    """Compare freshly computed rows with logged rows. Returns (largest difference, ok, details)."""
    worst, details = 0.0, []
    for row in new_rows:
        old = old_by_relation.get(row["relation"])
        if old is None:
            details.append((row["relation"], "missing in log", float("inf")))
            worst = float("inf")
            continue
        for col in CHECK_COLS:
            diff = abs(float(row[col]) - float(old[col]))
            worst = max(worst, diff)
            if diff > tol:
                details.append((row["relation"], col, diff))
    return worst, worst <= tol, details
