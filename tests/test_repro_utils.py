import csv

from src.eval.repro_utils import compare_rows, latest_logged_rows, sha256_file

FIELDS = ["run_id", "date", "owner", "model", "config", "seed", "relation",
          "MRR", "Hits@1", "Hits@3", "Hits@10", "AUC", "notes"]


def _row(run_id, config, seed, rel, mrr):
    return dict(run_id=run_id, date="2026-10-20", owner="P4", model="m", config=config, seed=seed,
                relation=rel, MRR=mrr, **{"Hits@1": "0.1", "Hits@3": "0.2", "Hits@10": "0.3", "AUC": "0.8"}, notes="")


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_latest_logged_rows_picks_newest_run_for_same_config_and_seed(tmp_path):
    p = tmp_path / "results.csv"
    rows = []
    for rid, mrr in [("rotate-s42-20261018100000", "0.10"), ("rotate-s42-20261019100000", "0.20")]:
        rows += [_row(rid, "configs/rotate.yaml", 42, "overall", mrr), _row(rid, "configs/rotate.yaml", 42, "Damage", mrr)]
    rows.append(_row("rotate-s7-20261020100000", "configs/rotate.yaml", 7, "overall", "0.99"))       # other seed
    rows.append(_row("popularity-s42-20261020100000", "configs/popularity.yaml", 42, "overall", "0.50"))  # other config
    rows.append(_row("legacy_run", "configs/rotate.yaml", 42, "all", "0.01"))                       # legacy: no 'overall'
    _write(p, rows)
    got = latest_logged_rows(p, "configs\\rotate.yaml", 42)                                          # backslashes are normalised
    assert got["overall"]["MRR"] == "0.20" and set(got) == {"overall", "Damage"}
    assert latest_logged_rows(p, "configs/vrgae.yaml", 42) == {}


def test_compare_rows_tolerance():
    old = {"overall": _row("a", "c", 42, "overall", "0.2000")}
    same = [_row("b", "c", 42, "overall", "0.2010")]
    worst, ok, details = compare_rows(same, old, tol=0.005)
    assert ok and details == [] and abs(worst - 0.001) < 1e-9
    worst, ok, details = compare_rows([_row("b", "c", 42, "overall", "0.3000")], old, tol=0.005)
    assert not ok and details[0][:2] == ("overall", "MRR")
    worst, ok, _ = compare_rows([_row("b", "c", 42, "Damage", "0.2")], old)
    assert not ok                                                    # relation missing in the log


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
