"""Build comparison tables from the newest complete protocol-v2 run per config.

Historical and legacy-unverified runs remain in the CSV, but are not eligible
for the headline table. Usage:
python src/eval/make_table.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "results.csv"
OUT = ROOT / "results" / "final_table.md"

NAMES = {
    "popularity": "Popularity (sanity check)",
    "rotate": "RotatE alone (baseline 1)",
    "rgcn_distmult": "R-GCN + DistMult (baseline 2)",
    "distmult_nograph": "DistMult, no graph (ablation)",
    "vrgae": "VRGAE-lite (ours, shared training recipe)",
    "vrgae_checkpoint": "VRGAE-lite (ours, as trained by Person 3)",
}
RELS = ["RegionIs", "ConditionIs", "Damage", "InflictHarmPart", "ManifestAs", "TemperIs"]


def load_latest() -> dict:
    """Select each config's newest complete protocol-v2 run."""
    runs = {}  # run_id -> {relation: row}
    with open(CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs.setdefault(row["run_id"], {})[row["relation"]] = row
    best = {}  # config stem -> (timestamp, run_id, rows)
    for rid, rows in runs.items():
        first = next(iter(rows.values()))
        if "overall" not in rows or first.get("provenance_status") != "complete":
            continue

        stem = Path(first["config"].replace("\\", "/")).stem
        ts = rid.rsplit("-", 1)[-1]
        protocol = first.get("protocol", "")
        if "v2" not in protocol and "protocol=v2" not in first.get("notes", ""):
            continue
        if stem not in best or ts > best[stem][0]:
            best[stem] = (ts, rid, rows)
    return best


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    best = load_latest()
    stems = [s for s in NAMES if s in best] + [s for s in best if s not in NAMES]
    name = lambda s: NAMES.get(s, s)
    seeds = sorted({best[s][2]["overall"]["seed"] for s in stems})

    t1 = [[name(s)] + [best[s][2]["overall"][k] for k in ("MRR", "Hits@1", "Hits@3", "Hits@10", "AUC")]
          + [best[s][2]["macro"]["MRR"], best[s][2]["macro"]["Hits@10"]] for s in stems]
    rel_hdr = [r + ("*" if r == "TemperIs" else "") for r in RELS]
    t2 = [[name(s)] + [best[s][2][r]["MRR"] for r in RELS] for s in stems]
    t3 = [[name(s)] + [best[s][2][r]["Hits@10"] for r in RELS] for s in stems]
    n_row = ["n (test facts)"] + [best[stems[0]][2][r]["notes"].split(";")[0].replace("n=", "") for r in RELS]

    text = "\n\n".join([
        "## Table 1: overall and macro-average (test split)",
        md(["Model", "MRR", "Hits@1", "Hits@3", "Hits@10", "AUC", "Macro MRR", "Macro Hits@10"], t1),
        "## Table 2: MRR per relation (test split)",
        md(["Model"] + rel_hdr, t2 + [n_row]),
        "## Table 3: Hits@10 per relation (test split)",
        md(["Model"] + rel_hdr, t3 + [n_row]),
        f"Protocol: v2 (shared filtered negative sampler and evaluator). Seed(s): {', '.join(seeds)}"
        " "
        "(single seed: treat MRR differences below about 0.01 as ties). "
        "Best epoch chosen on the full validation split; all numbers are filtered tail prediction on test. "
        "Earlier protocol-v1 rows are retained in results.csv for history but excluded from this comparison.\n\n"
        "Only complete-provenance protocol-v2 runs are included; historical unverified "
        "rows remain in results.csv but are excluded. See results/REPRODUCE.md for the rerun audit.\n\n"
        "\\* TemperIs has only 9 test facts and is not reliable.\n\n"
        "AUC is secondary: many fixed negatives have the wrong entity type, so AUC rewards type-matching "
        "more than fine-grained ranking. Rely on MRR and Hits@K.",
    ])
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()