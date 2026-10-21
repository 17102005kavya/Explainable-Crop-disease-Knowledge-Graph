"""Verify provenance and optionally rerun one recorded experiment."""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.eval.provenance import (
    data_fingerprint,
    environment_identity,
    git_identity,
    protocol_name,
    sha256_file,
    source_fingerprint,
)


ROOT = Path(__file__).resolve().parent
PROVENANCE_FIELDS = (
    "protocol",
    "source_commit",
    "source_dirty",
    "source_sha256",
    "config_sha256",
    "effective_config_json",
    "data_sha256",
    "python_version",
    "platform",
    "torch_version",
    "torch_cuda_version",
    "torch_geometric_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "scikit_learn_version",
    "pyyaml_version",
    "checkpoint_path",
    "checkpoint_sha256",
    "source_checkpoint_path",
    "source_checkpoint_sha256",
    "selected_epoch",
    "provenance_status",
)
METRICS = ("MRR", "Hits@1", "Hits@3", "Hits@10", "AUC")


def read_run(csv_path: Path, run_id: str) -> dict[str, dict]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["run_id"] == run_id]
    if not rows:
        raise ValueError(f"Run {run_id!r} was not found in {csv_path}")
    by_relation = {row["relation"]: row for row in rows}
    if len(by_relation) != len(rows):
        raise ValueError(f"Run {run_id!r} has duplicate relation rows")
    return by_relation


def expected_environment(config: dict, config_path: Path) -> dict[str, str]:
    commit, dirty = git_identity()
    data_dir = Path(config.get("data_dir", "data/processed"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return {
        "protocol": protocol_name(config),
        "source_commit": commit,
        "source_dirty": dirty,
        "source_sha256": source_fingerprint(),
        "config_sha256": sha256_file(config_path),
        "data_sha256": data_fingerprint(data_dir.resolve()),
        **environment_identity(),
    }


def verify_provenance(run: dict[str, dict]) -> tuple[Path, dict]:
    first = next(iter(run.values()))
    missing = [field for field in PROVENANCE_FIELDS if not first.get(field)]
    if missing or first.get("provenance_status") != "complete":
        raise ValueError(
            "This run is recorded but cannot be reproduced from complete verified "
            f"provenance (status={first.get('provenance_status') or 'missing'}). "
            f"Missing fields: {', '.join(missing) or 'complete provenance status'}."
        )
    if any(row.get(field) != first.get(field) for row in run.values()
           for field in PROVENANCE_FIELDS):
        raise ValueError("Rows in the selected run contain inconsistent provenance fields")

    config_path = Path(first["config"])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Recorded config does not exist: {config_path}")
    try:
        config = json.loads(first["effective_config_json"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("The recorded effective config is unavailable or invalid") from error
    current = expected_environment(config, config_path)
    differences = [
        f"{key}: recorded={first.get(key)!r}, current={value!r}"
        for key, value in current.items()
        if key not in ("source_commit", "source_dirty") and first.get(key) != value
    ]
    checkpoint_path = Path(first["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        differences.append(f"checkpoint missing: {checkpoint_path}")
    elif sha256_file(checkpoint_path) != first["checkpoint_sha256"]:
        differences.append(f"checkpoint checksum changed: {checkpoint_path}")

    source_checkpoint = first.get("source_checkpoint_path")
    if source_checkpoint and source_checkpoint != "not-applicable":
        source_checkpoint_path = Path(source_checkpoint)
        if not source_checkpoint_path.is_absolute():
            source_checkpoint_path = ROOT / source_checkpoint_path
        if (not source_checkpoint_path.is_file()
                or sha256_file(source_checkpoint_path) != first["source_checkpoint_sha256"]):
            differences.append(f"source checkpoint missing or changed: {source_checkpoint_path}")
    if differences:
        raise ValueError("Provenance does not match this checkout:\n- " + "\n- ".join(differences))
    return config_path, config


def compare_metrics(expected: dict[str, dict], actual: dict[str, dict], tolerance: float) -> list[str]:
    mismatches = []
    if set(expected) != set(actual):
        return [f"relation rows differ: expected {sorted(expected)}, got {sorted(actual)}"]
    for relation in expected:
        expected_epoch = expected[relation].get("selected_epoch", "")
        actual_epoch = actual[relation].get("selected_epoch", "")
        if expected_epoch != actual_epoch:
            mismatches.append(
                f"{relation} selected_epoch: expected {expected_epoch!r}, got {actual_epoch!r}"
            )
        for metric in METRICS:
            delta = abs(float(expected[relation][metric]) - float(actual[relation][metric]))
            if delta > tolerance:
                mismatches.append(
                    f"{relation} {metric}: expected {expected[relation][metric]}, "
                    f"got {actual[relation][metric]} (delta {delta:.6f})"
                )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="run_id from results/results.csv")
    parser.add_argument("--csv", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--verify-only", action="store_true",
                        help="validate recorded provenance without rerunning training")
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be non-negative")
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    expected = read_run(csv_path, args.run_id)
    config_path, config = verify_provenance(expected)
    if args.verify_only:
        print(f"Verified provenance for run {args.run_id}")
        return 0

    with tempfile.TemporaryDirectory(prefix="cropdp-reproduce-") as temp:
        temp_dir = Path(temp)
        output_csv = temp_dir / "results.csv"
        checkpoint_dir = temp_dir / "checkpoints"
        command = [
            sys.executable,
            str(ROOT / "src" / "eval" / "run_experiment.py"),
            "--config",
            str(config_path),
            "--output-csv",
            str(output_csv),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        import yaml

        yaml_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if yaml_config.get("model") != config.get("model"):
            command.extend(["--model", config["model"]])
        subprocess.run(command, cwd=ROOT, check=True)
        actual_run_ids = read_csv_run_id(output_csv)
        if len(actual_run_ids) != 1:
            raise ValueError(f"Expected one run in reproduction output, got {sorted(actual_run_ids)}")
        actual = read_run(output_csv, next(iter(actual_run_ids)))

    mismatches = compare_metrics(expected, actual, args.tolerance)
    if mismatches:
        print("Reproduction check FAILED:")
        print("\n".join(f"- {item}" for item in mismatches))
        return 1
    print(f"Reproduction check passed for {args.run_id} (tolerance={args.tolerance})")
    return 0


def read_csv_run_id(csv_path: Path) -> set[str]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        return {row["run_id"] for row in csv.DictReader(stream)}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise SystemExit(f"Reproduction check cannot run: {error}") from error
