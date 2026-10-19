"""Owner: Person 4 (Part D). Run ONE config, evaluate on TEST through evaluate_ranking / evaluate_auc,
append rows to results/results.csv (one row per relation + overall + macro).

Usage (from the repo root):
    python src/eval/run_experiment.py --config configs/rotate.yaml
Validation is used only to pick the best epoch; the logged numbers are TEST numbers.
"""
import argparse
import csv
import datetime as dt
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

FIELDS = ["run_id", "date", "owner", "model", "config", "seed", "relation",
          "MRR", "Hits@1", "Hits@3", "Hits@10", "AUC", "notes"]
RESULTS = ROOT / "results" / "results.csv"


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


def make_rows(cfg, cfg_path, d, res, auc, extra):
    names = d["meta"]["relations"]
    now = dt.datetime.now()
    run_id = f"{Path(cfg_path).stem}-s{cfg['seed']}-{now:%Y%m%d%H%M%S}"
    base = dict(run_id=run_id, date=now.date().isoformat(), owner=cfg.get("owner", "P4"),
                model=cfg["model"], config=str(cfg_path), seed=cfg["seed"])
    info = "; ".join(f"{k}={v}" for k, v in extra.items())
    if cfg.get("notes"):
        info = (info + "; " if info else "") + str(cfg["notes"])

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="optional override of model in the YAML")
    ap.add_argument("--no_log", action="store_true", help="evaluate but do not write results.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    print("CONFIG:", cfg)
    d = load_processed(str(ROOT / cfg.get("data_dir", "data/processed")))

    model, extra = build_and_fit(cfg, d)
    res, auc = evaluate_test(model, d)
    rows = make_rows(cfg, args.config, d, res, auc, extra)

    print(f"\n=== {cfg['model']} | TEST ===")
    for r in rows:
        print(f"{r['relation']:16s} MRR={r['MRR']} H@1={r['Hits@1']} H@3={r['Hits@3']} "
              f"H@10={r['Hits@10']} AUC={r['AUC']}  [{r['notes']}]")
    if not args.no_log:
        RESULTS.parent.mkdir(exist_ok=True)
        new = (not RESULTS.exists()) or RESULTS.stat().st_size == 0
        with open(RESULTS, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\nappended {len(rows)} rows to {RESULTS}")


if __name__ == "__main__":
    main()