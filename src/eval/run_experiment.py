"""Owner: Person 4 (Part D). Run ONE config, evaluate on TEST through evaluate_ranking / evaluate_auc,
append rows to results/results.csv (one row per relation + overall + macro).

Usage (from the repo root):
    python src/eval/run_experiment.py --config configs/rotate.yaml
Validation is used only to pick the best epoch; the logged numbers are TEST numbers.
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_processed                                   # noqa: E402
from src.eval.baselines import (PopularityBaseline, RotatEBaseline,           # noqa: E402
                                RGCNDistMultBaseline)
from src.eval.evaluator import evaluate_ranking, evaluate_auc                 # noqa: E402
from src.eval.repro_utils import compare_rows, git_commit, latest_logged_rows   # noqa: E402
from src.eval.provenance import (data_fingerprint, environment_identity,      # noqa: E402
                                git_identity, protocol_name, sha256_file,
                                source_fingerprint)

FIELDS = ["run_id", "date", "owner", "model", "config", "seed", "relation",
          "MRR", "Hits@1", "Hits@3", "Hits@10", "AUC", "notes", "protocol",
          "source_commit", "source_dirty", "source_sha256", "config_sha256",
          "effective_config_json", "data_sha256", "python_version", "platform", "torch_version",
          "torch_cuda_version", "torch_geometric_version", "numpy_version",
          "pandas_version", "scipy_version", "scikit_learn_version",
          "pyyaml_version", "checkpoint_path", "checkpoint_sha256",
          "source_checkpoint_path", "source_checkpoint_sha256",
          "selected_epoch", "provenance_status"]
RESULTS = ROOT / "results" / "results.csv"
PROTOCOL = "v2"   # v2 = shared UniformNegativeSampler + shared evaluator (average ties) + full-val epoch selection


def build_and_fit(cfg: dict, d: dict):
    name, p, tr = cfg["model"], cfg.get("params", {}), cfg.get("training", {})
    N, R = d["meta"]["num_entities"], len(d["meta"]["relations"])
    seed = cfg["seed"]
    torch.manual_seed(seed)
    if name == "vrgae":
        from src.eval.vrgae_adapter import build_vrgae      # lazy import: needs the real latent.py / loss.py
        return build_vrgae(cfg, d)                      # seeds the model initialisation too
    if name == "popularity":
        return PopularityBaseline(d["train"], N, R), {}
    if name == "rotate":
        model = RotatEBaseline(N, R, **p)
    elif name == "rgcn_distmult":
        model = RGCNDistMultBaseline(d, **p)
    else:
        raise SystemExit(f"model '{name}' is not supported yet (vrgae is plugged in later)")
    best_ep, best_val = model.fit(d, seed=seed, **tr)
    return model, {"best_epoch": best_ep, "val_MRR": round(best_val, 4)}


@torch.no_grad()
def evaluate_test(model, d: dict):
    N = d["meta"]["num_entities"]
    known = torch.cat([d["train"], d["val"], d["test"]])
    pos, neg = d["test"], d["test_neg"]
    res = evaluate_ranking(model.make_score_all_tails(), pos, known, N)
    auc = {"overall": evaluate_auc(model.score_triples, pos, neg)}
    for r in range(len(d["meta"]["relations"])):
        m = pos[:, 1] == r                        # negatives are row-aligned with positives
        auc[r] = evaluate_auc(model.score_triples, pos[m], neg[m])
    return res, auc


def make_rows(cfg, cfg_path, d, res, auc, extra, provenance):
    names = d["meta"]["relations"]
    now = provenance["started_at"]
    run_id = provenance["run_id"]
    base = dict(run_id=run_id, date=now.date().isoformat(), owner=cfg.get("owner", "P4"),
                model=cfg["model"], config=str(cfg_path), seed=cfg["seed"],
                **{key: value for key, value in provenance.items()
                   if key not in ("run_id", "started_at")})
    info = "; ".join(f"{k}={v}" for k, v in extra.items())
    if cfg.get("notes"):
        info = (info + "; " if info else "") + str(cfg["notes"])
    info = (info + "; " if info else "") + f"protocol={PROTOCOL}; torch={torch.__version__}; commit={git_commit(ROOT)}"

    def row(rel, g, a, n_txt, note=""):
        notes = "; ".join(x for x in [n_txt, note, info] if x)
        return {**base, "relation": rel, "MRR": f"{res[g + 'MRR']:.4f}",
                "Hits@1": f"{res[g + 'Hits@1']:.4f}", "Hits@3": f"{res[g + 'Hits@3']:.4f}",
                "Hits@10": f"{res[g + 'Hits@10']:.4f}", "AUC": f"{a:.4f}", "notes": notes}

    rows = [row("overall", "", auc["overall"], f"n={int(res['n'])}", "micro average, test split")]
    macro_auc = sum(auc[r] for r in range(len(names))) / len(names)
    rows.append(row("macro", "macro/", macro_auc, "mean over relations", "AUC = mean of per-relation AUC"))
    for r, nm in enumerate(names):
        n = int(res[f"r{r}/n"])
        warn = "UNRELIABLE (very few facts)" if n < 30 else ""
        rows.append(row(nm, f"r{r}/", auc[r], f"n={n}", warn))
    return rows


def save_checkpoint(model, checkpoint_path, cfg, extra, provenance):
    checkpoint_model = getattr(model, "model", model)
    if isinstance(checkpoint_model, torch.nn.Module):
        state = checkpoint_model.state_dict()
    elif hasattr(checkpoint_model, "counts"):
        state = {"counts": checkpoint_model.counts}
    else:
        raise TypeError(f"Cannot save checkpoint for model type {type(checkpoint_model).__name__}")
    payload = {
        "model_state": state,
        "config": cfg,
        "training": extra,
        "provenance": {k: v for k, v in provenance.items()
                       if k not in ("started_at",)},
    }
    if hasattr(checkpoint_model, "N"):
        payload["num_entities"] = checkpoint_model.N
    if cfg["model"] == "popularity":
        payload["kind"] = "popularity"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def ensure_results_schema(path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        existing_fields = reader.fieldnames or []
        old_rows = list(reader)
    if existing_fields == FIELDS:
        return
    migrated_rows = []
    for row in old_rows:
        migrated = {field: row.get(field, "") for field in FIELDS}
        if not migrated.get("provenance_status"):
            migrated["provenance_status"] = "legacy-unverified"
        migrated_rows.append(migrated)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(migrated_rows)
    temp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="optional override of model in the YAML")
    ap.add_argument("--no_log", action="store_true", help="evaluate but do not write results.csv")
    ap.add_argument("--check", action="store_true",
                    help="re-run and compare with the latest logged run of this config; writes nothing")
    ap.add_argument("--tol", type=float, default=0.005, help="max allowed difference for --check")
    ap.add_argument("--output-csv", type=Path, default=RESULTS,
                    help="CSV destination (defaults to results/results.csv)")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "results" / "checkpoints",
                    help="directory for this run's model checkpoint")
    args = ap.parse_args()

    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = ROOT / config_file
    config_file = config_file.resolve()
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    if cfg.get("source") == "checkpoint":
        input_checkpoint = Path(cfg["checkpoint"])
        if not input_checkpoint.is_absolute():
            cfg["checkpoint"] = str((ROOT / input_checkpoint).resolve())
    print("CONFIG:", cfg)
    data_dir = Path(cfg.get("data_dir", "data/processed"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    data_dir = data_dir.resolve()
    d = load_processed(str(data_dir))

    started_at = dt.datetime.now()
    model, extra = build_and_fit(cfg, d)
    run_id = f"{config_file.stem}-s{cfg['seed']}-{started_at:%Y%m%d%H%M%S}"
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    checkpoint_path = (checkpoint_dir / f"{run_id}.pt").resolve()
    source_checkpoint_path = "not-applicable"
    source_checkpoint_sha256 = ""
    if cfg.get("source") == "checkpoint":
        source_checkpoint_path = Path(cfg["checkpoint"])
        if not source_checkpoint_path.is_absolute():
            source_checkpoint_path = ROOT / source_checkpoint_path
        source_checkpoint_path = source_checkpoint_path.resolve()
        source_checkpoint_sha256 = sha256_file(source_checkpoint_path)
        try:
            source_checkpoint_path = source_checkpoint_path.relative_to(ROOT).as_posix()
        except ValueError:
            source_checkpoint_path = str(source_checkpoint_path)

    source_commit, source_dirty = git_identity()
    try:
        config_name = config_file.relative_to(ROOT).as_posix()
    except ValueError:
        config_name = str(config_file)
    try:
        checkpoint_name = checkpoint_path.relative_to(ROOT).as_posix()
    except ValueError:
        checkpoint_name = str(checkpoint_path)
    provenance = {
        "run_id": run_id,
        "started_at": started_at,
        "protocol": protocol_name(cfg),
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_sha256": source_fingerprint(),
        "config_sha256": sha256_file(config_file),
        "effective_config_json": json.dumps(cfg, sort_keys=True, separators=(",", ":")),
        "data_sha256": data_fingerprint(data_dir),
        **environment_identity(),
        "checkpoint_path": checkpoint_name,
        "checkpoint_sha256": "",
        "source_checkpoint_path": source_checkpoint_path,
        "source_checkpoint_sha256": source_checkpoint_sha256 or "not-applicable",
        "selected_epoch": extra.get("best_epoch") or extra.get("ckpt_epoch") or "not-applicable",
        "provenance_status": "complete",
    }
    save_checkpoint(model, checkpoint_path, cfg, extra, provenance)
    provenance["checkpoint_sha256"] = sha256_file(checkpoint_path)
    res, auc = evaluate_test(model, d)
    rows = make_rows(cfg, config_name, d, res, auc, extra, provenance)

    print(f"\n=== {cfg['model']} | TEST ===")
    for r in rows:
        print(f"{r['relation']:16s} MRR={r['MRR']} H@1={r['Hits@1']} H@3={r['Hits@3']} "
              f"H@10={r['Hits@10']} AUC={r['AUC']}  [{r['notes']}]")
    if args.check:
        old = latest_logged_rows(RESULTS, args.config, cfg["seed"])
        if not old:
            print(f"\nCHECK: no logged run for {args.config} (seed {cfg['seed']}); run it once without --check")
            sys.exit(2)
        worst, ok, details = compare_rows(rows, old, args.tol)
        print(f"\nCHECK {'PASS' if ok else 'FAIL'}: largest difference to the logged run = {worst:.5f} "
              f"(tolerance {args.tol})")
        for rel, col, diff in details[:10]:
            print(f"  {rel:16s} {col:14s} diff={diff}")
        sys.exit(0 if ok else 1)
    if not args.no_log:
        output_csv = args.output_csv
        if not output_csv.is_absolute():
            output_csv = ROOT / output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        ensure_results_schema(output_csv)
        new = (not output_csv.exists()) or output_csv.stat().st_size == 0
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()