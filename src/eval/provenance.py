"""Helpers for identifying the source, inputs, and environment of an experiment."""
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = (
    "meta.json",
    "entities.csv",
    "relations.csv",
    "triples_clean.csv",
    "provenance.csv",
    "conflicts.csv",
    "train.npy",
    "val.npy",
    "test.npy",
    "node_types.npy",
    "graph_edge_index.npy",
    "graph_edge_type.npy",
    "val_neg.npy",
    "test_neg.npy",
)
PACKAGE_DISTRIBUTIONS = {
    "torch_version": "torch",
    "torch_geometric_version": "torch-geometric",
    "numpy_version": "numpy",
    "pandas_version": "pandas",
    "scipy_version": "scipy",
    "scikit_learn_version": "scikit-learn",
    "pyyaml_version": "PyYAML",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def data_fingerprint(data_dir: Path) -> str:
    missing = [name for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Processed dataset is incomplete at {data_dir}; missing: {', '.join(missing)}"
        )
    digest = hashlib.sha256()
    for name in DATA_FILES:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(sha256_file(data_dir / name)))
    return digest.hexdigest()


def source_fingerprint() -> str:
    paths = list((ROOT / "src").rglob("*.py"))
    paths += list((ROOT / "configs").rglob("*.yaml"))
    paths += [ROOT / name for name in ("requirements.txt", "requirements-lock.txt", "reproduce.py")]
    paths = [path for path in paths if path.is_file()]
    return _manifest_hash(paths)


def git_identity() -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs", "requirements.txt",
         "requirements-lock.txt", "reproduce.py"],
        cwd=ROOT,
        text=True,
    )
    return commit, str(bool(status.strip())).lower()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment_identity() -> dict[str, str]:
    versions = {key: package_version(name)
                for key, name in PACKAGE_DISTRIBUTIONS.items()}
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_version": str(__import__("torch").version.cuda or "cpu"),
        **versions,
    }


def protocol_name(config: dict) -> str:
    model = config["model"]
    if model == "popularity":
        return "popularity-sanity-v1"
    if model == "vrgae":
        if config.get("source", "train") == "checkpoint":
            return "vrgae-checkpoint-evaluation-v2"
        return "vrgae-shared-training-v2"
    if model == "rgcn_distmult" and not config.get("params", {}).get("use_graph", True):
        return "distmult-no-graph-shared-runner-v2"
    return f"{model}-shared-runner-v2"
