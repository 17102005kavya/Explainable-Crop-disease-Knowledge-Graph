#!/usr/bin/env python3
"""
Part A pipeline for CropDP-KG: clean -> check conflicts -> type entities -> split -> negatives.

Usage (from the repo root):
    python prepare_cropdp.py --raw data/raw --inspect          # look at the CSVs first
    python prepare_cropdp.py --raw data/raw --out data/processed --seed 42
    python prepare_cropdp.py --demo data/demo_raw              # write a tiny fake dataset to dry-run

Expected input: the relation*.csv files from the CropDP-KG Figshare record
(relationEng, relationarea, relationcon, relationcrop, relationpart, relationsym, relationtem).
Each row is a triple. The first three columns are read as head, relation, tail; if a file only has two
columns, the relation is inferred from the file name. Use --no-header if the files have no header row.

Outputs in --out:
    entities.csv, relations.csv, triples_clean.csv      readable ID tables
    train/val/test .npy (+ .pt if torch is installed)   int64 arrays of shape [M, 3] = (head, relation, tail)
    node_types.npy                                       multi-hot entity-type matrix [num_entities, 8]
    graph.npy / graph.pt                                 train-only edge_index and edge_type incl. inverse edges
    val_neg / test_neg                                   fixed negatives (1 per positive) for AUC
    conflicts.csv, provenance.csv, data_report.md, meta.json
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # the pipeline itself needs only numpy/pandas
    torch = None

# ----------------------------------------------------------------------------- constants
CANON_ORDER = ["RegionIs", "ConditionIs", "Damage", "EnglishName", "InflictHarmPart", "ManifestAs", "TemperIs"]
CANON_REL = {r.lower(): r for r in CANON_ORDER}
# type of the TAIL entity for each relation (the head is always a pest/disease name)
TAIL_TYPE = {"RegionIs": "Area", "ConditionIs": "Condition", "Damage": "Crop", "EnglishName": "EnglishName",
             "InflictHarmPart": "Part", "ManifestAs": "Symptom", "TemperIs": "Temperature"}
HEAD_TYPE = "PestDisease"
TYPES = [HEAD_TYPE, "Crop", "Symptom", "Condition", "Part", "Area", "Temperature", "EnglishName"]
FILE_HINT = {"relationeng": "EnglishName", "relationarea": "RegionIs", "relationcon": "ConditionIs",
             "relationcrop": "Damage", "relationpart": "InflictHarmPart", "relationsym": "ManifestAs",
             "relationtem": "TemperIs"}
FUNCTIONAL = ["EnglishName", "TemperIs"]  # expected to have one tail per head; more than one is flagged
MISSING = {"", "nan", "none", "null", "n/a", "na", "-", "--", "无", "暂无", "未知"}


# ----------------------------------------------------------------------------- text helpers
def norm(s) -> str:
    """NFKC (full-width -> half-width), drop zero-width chars, collapse whitespace, strip quotes."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("\"'\u201c\u201d\u2018\u2019").strip()


def near_key(s: str) -> str:
    """Conservative key for near-duplicate detection: case-fold and drop whitespace only."""
    return re.sub(r"\s+", "", s).casefold()


def canon_rel(raw, hint):
    k = re.sub(r"[^a-z]", "", norm(raw).lower())
    return CANON_REL.get(k) or hint or norm(raw)


# ----------------------------------------------------------------------------- load
def read_csv_any(path, header):
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, header=0 if header else None, dtype=str, encoding=enc, keep_default_na=False)
        except UnicodeDecodeError as e:
            last = e
    raise last


def find_files(raw_dir):
    return sorted(p for p in Path(raw_dir).rglob("*.csv")
                  if p.stem.lower().replace(" ", "").startswith("relation"))


def inspect(raw_dir, header=True):
    for p in find_files(raw_dir):
        df = read_csv_any(p, header)
        print(f"\n== {p.name}: {len(df)} rows, columns = {list(df.columns)}")
        print(df.head(3).to_string())


def _is_index_col(col: pd.Series, name) -> bool:
    """Leading row-index column: digits only and unique per row (or a typical index header)."""
    n = str(name).strip().lower()
    if n.startswith("unnamed") or n in ("", "index", "id", "idx"):
        return True
    v = col.astype(str).str.strip()
    return bool(len(v) > 0 and v.str.fullmatch(r"\d+").all() and v.nunique() == len(v))


def _relation_like(col: pd.Series) -> bool:
    keys = col.astype(str).map(lambda x: re.sub(r"[^a-z]", "", norm(x).lower()))
    return bool(len(keys) > 0 and keys.isin(CANON_REL).mean() >= 0.5)


def pick_columns(df):
    """Choose head / relation / tail columns by header name (CropDP-KG layout):
    head = 'Chinese entity name'; relation = 'English relation'; tail = the first remaining 'Chinese <x>' column
    (or 'tem'), skipping scientific-name, English-translation, relation and index columns."""
    cols = list(df.columns)
    low = {c: str(c).strip().lower() for c in cols}
    head = next((c for c in cols if low[c] == "chinese entity name"), None)
    rel = next((c for c in cols if low[c] == "english relation"), None) or \
        next((c for c in cols if "relation" in low[c]), None)
    def bad(c):
        return (c == head or c == rel or low[c].startswith("english") or
                any(w in low[c] for w in ("scientific", "binomial", "relation", "number", "unnamed", "index")))
    cand = sorted((c for c in cols if not bad(c)), key=lambda c: 0 if low[c].startswith("chinese") else 1)
    return head, rel, (cand[0] if cand else None)


def load_raw(raw_dir, header=True) -> pd.DataFrame:
    rows = []
    files = find_files(raw_dir)
    if not files:
        raise SystemExit(f"No relation*.csv files found in {raw_dir}")
    for p in files:
        stem = p.stem.lower().replace(" ", "")
        hint = next((v for k, v in FILE_HINT.items() if stem.startswith(k)), None)
        df = read_csv_any(p, header)
        notes = []
        if df.shape[1] >= 3 and _is_index_col(df.iloc[:, 0], df.columns[0]):
            notes.append(f"dropped index column {df.columns[0]!r}")
            df = df.iloc[:, 1:]
        h_col, r_col, t_col = pick_columns(df) if header else (None, None, None)
        t_en = None
        if t_col is not None and str(t_col).strip().lower().startswith("chinese "):
            en_name = "english " + str(t_col).strip().lower()[len("chinese "):]
            en_col = next((c for c in df.columns if str(c).strip().lower() == en_name), None)
            if en_col is not None:
                t_en = df[en_col]
        if h_col is not None and t_col is not None:
            h, t = df[h_col], df[t_col]
            if r_col is not None and _relation_like(df[r_col]):
                r = df[r_col]
                notes.append(f"head={h_col!r} relation={r_col!r} tail={t_col!r}")
            else:
                r = pd.Series([""] * len(df), index=df.index)
                notes.append(f"head={h_col!r} tail={t_col!r}, relation from file name -> {hint}")
        elif df.shape[1] >= 3 and _relation_like(df.iloc[:, 1]):
            h, r, t = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
            notes.append(f"head={df.columns[0]!r} relation={df.columns[1]!r} tail={df.columns[2]!r}")
        elif df.shape[1] >= 2:
            h, t, r = df.iloc[:, 0], df.iloc[:, 1], pd.Series([""] * len(df), index=df.index)
            notes.append(f"head={df.columns[0]!r} tail={df.columns[1]!r}, relation from file name -> {hint}")
            if df.shape[1] > 2:
                notes.append(f"[check] {df.shape[1] - 2} extra column(s) ignored: {list(df.columns[2:])}")
        else:
            print(f"[warn] {p.name}: fewer than 2 usable columns, skipped")
            continue
        print(f"[load] {p.name}: {len(df)} rows; " + "; ".join(notes))
        first_line = 2 if header else 1
        te = list(t_en) if t_en is not None else [""] * len(df)
        for i, (a, b, c, e) in enumerate(zip(h, r, t, te)):
            rows.append((p.name, i + first_line, a, b, c, hint, e))
    return pd.DataFrame(rows, columns=["file", "line", "raw_h", "raw_r", "raw_t", "hint", "raw_t_en"])


# ----------------------------------------------------------------------------- clean + conflicts
def clean(raw: pd.DataFrame, merge_near_dups=False):
    df = raw.copy()
    df["h"] = df.raw_h.map(norm)
    df["t"] = df.raw_t.map(norm)
    df["r"] = [canon_rel(a, b) for a, b in zip(df.raw_r, df.hint)]
    df["status"] = "kept"

    miss = df.h.str.lower().isin(MISSING) | df.t.str.lower().isin(MISSING) | (df.r == "")
    df.loc[miss, "status"] = "dropped_missing"

    # near-duplicate entity strings (differ only by case/spacing after NFKC)
    kept = df[df.status == "kept"]
    counts = pd.concat([kept.h, kept.t]).value_counts()
    groups = {}
    for s, c in counts.items():
        groups.setdefault(near_key(s), []).append((s, c))
    near = {k: v for k, v in groups.items() if len(v) > 1}
    remap = {}
    for k, v in near.items():
        best = sorted(v, key=lambda x: (-x[1], len(x[0]), x[0]))[0][0]
        for s, _ in v:
            remap[s] = best
    if merge_near_dups and remap:
        df["h"] = df.h.map(lambda s: remap.get(s, s))
        df["t"] = df.t.map(lambda s: remap.get(s, s))

    ok = df.status == "kept"
    loop = ok & (df.h == df.t)
    df.loc[loop, "status"] = "dropped_selfloop"

    ok = df.status == "kept"
    dup = ok & df.duplicated(["h", "r", "t"], keep="first")
    df.loc[dup, "status"] = "dropped_duplicate"
    return df, near


def find_conflicts(kept: pd.DataFrame, near: dict, merged: bool) -> pd.DataFrame:
    out = []
    # 1) functional relations with several tails for one head
    for rel in FUNCTIONAL:
        sub = kept[kept.r == rel]
        g = sub.groupby("h").t.nunique()
        for head in g[g > 1].index:
            tails = sorted(sub[sub.h == head].t.unique())
            out.append(("multi_valued_functional", rel, head, "", " | ".join(tails)))
    # 2) same (head, tail) pair under more than one relation
    g = kept.groupby(["h", "t"]).r.nunique()
    for (h, t) in g[g > 1].index:
        rels = sorted(kept[(kept.h == h) & (kept.t == t)].r.unique())
        out.append(("pair_in_multiple_relations", "", h, t, " | ".join(rels)))
    # 3) a string used as a pest/disease head AND as a tail
    both = set(kept.h) & set(kept.t)
    for s in sorted(both):
        rels = sorted(kept[kept.t == s].r.unique())
        out.append(("head_and_tail_type_clash", "", s, "", "tail of: " + " | ".join(rels)))
    # 4) a tail string used under several relations (several entity types)
    tt = kept.groupby("t").r.nunique()
    for s in tt[tt > 1].index:
        rels = sorted(kept[kept.t == s].r.unique())
        out.append(("tail_in_multiple_types", "", "", s, " | ".join(rels)))
    # 5) near-duplicate entity strings
    for k, v in near.items():
        out.append(("near_duplicate_entities", "", "", "", " | ".join(s for s, _ in v)
                    + ("  (merged)" if merged else "  (NOT merged)")))
    return pd.DataFrame(out, columns=["kind", "relation", "head", "tail", "details"])


def english_map(df):
    """Chinese entity -> most frequent English translation (pest names from EnglishName rows, tails from the
    'English <x>' columns). Only used to make entities.csv readable; it does not affect any ID."""
    from collections import Counter
    cnt = {}
    used = df[df.status.isin(["kept", "dropped_by_flag"])]
    for h, t in used.loc[used.r == "EnglishName", ["h", "t"]].itertuples(index=False):
        cnt.setdefault(h, Counter())[t] += 1
    en = used.raw_t_en.map(norm)
    for t, e in zip(used.t[en != ""], en[en != ""]):
        cnt.setdefault(t, Counter())[e] += 1
    return {k: v.most_common(1)[0][0] for k, v in cnt.items()}


# ----------------------------------------------------------------------------- ids, types
def build_ids(kept: pd.DataFrame):
    ents = sorted(set(kept.h) | set(kept.t))
    e2i = {e: i for i, e in enumerate(ents)}
    present = set(kept.r)
    rels = [r for r in CANON_ORDER if r in present] + sorted(present - set(CANON_ORDER))
    r2i = {r: i for i, r in enumerate(rels)}
    tri = np.stack([kept.h.map(e2i).to_numpy(), kept.r.map(r2i).to_numpy(), kept.t.map(e2i).to_numpy()], 1).astype(np.int64)
    types = np.zeros((len(ents), len(TYPES)), dtype=np.float32)
    for h, r, t in tri:
        types[h, TYPES.index(HEAD_TYPE)] = 1
        types[t, TYPES.index(TAIL_TYPE.get(rels[r], "Condition"))] = 1
    return ents, rels, e2i, r2i, tri, types


# ----------------------------------------------------------------------------- split
def make_splits(tri, n_ent, val_frac, test_frac, seed):
    """Entity-covering split: every entity keeps at least one train triple (transductive setting).
    Val/test are drawn per relation from triples that are not needed for coverage."""
    rng = np.random.default_rng(seed)
    N = len(tri)
    covered = np.zeros(n_ent, bool)
    # (head, tail) pairs that occur under several relations stay in train, otherwise a held-out triple
    # could be guessed from its sibling in train (pair leakage)
    pair = tri[:, 0].astype(np.int64) * n_ent + tri[:, 2]
    multi = (pd.DataFrame({"p": pair, "r": tri[:, 1]}).groupby("p").r.transform("nunique").to_numpy() > 1)
    forced = multi.copy()
    covered[tri[multi, 0]] = True
    covered[tri[multi, 2]] = True
    for i in rng.permutation(N):
        if forced[i]:
            continue
        h, _, t = tri[i]
        if not covered[h] or not covered[t]:
            forced[i] = True
            covered[h] = covered[t] = True
    split = np.zeros(N, np.int8)  # 0 train, 1 val, 2 test
    stats = {}
    for r in np.unique(tri[:, 1]):
        idx_all = np.where(tri[:, 1] == r)[0]
        idx = rng.permutation(idx_all[~forced[idx_all]])
        want_v, want_t = round(val_frac * len(idx_all)), round(test_frac * len(idx_all))
        if want_v + want_t > len(idx):  # not enough eligible triples: shrink proportionally
            tot = len(idx)
            want_v = int(tot * val_frac / (val_frac + test_frac))
            want_t = tot - want_v
        split[idx[:want_v]] = 1
        split[idx[want_v:want_v + want_t]] = 2
        stats[int(r)] = dict(total=len(idx_all), eligible=len(idx), val_target=round(val_frac * len(idx_all)),
                             test_target=round(test_frac * len(idx_all)))
    return split, stats


def leakage_check(tri, split, n_ent, R):
    tr, va, te = tri[split == 0], tri[split == 1], tri[split == 2]
    key = lambda a: set(map(tuple, a.tolist()))
    train_ents = set(tr[:, 0].tolist()) | set(tr[:, 2].tolist())
    ktr, kva, kte = key(tr), key(va), key(te)
    res = {
        "train_val_overlap": len(ktr & kva),
        "train_test_overlap": len(ktr & kte),
        "val_test_overlap": len(kva & kte),
        "val_entities_missing_in_train": len((set(va[:, 0]) | set(va[:, 2])) - train_ents),
        "test_entities_missing_in_train": len((set(te[:, 0]) | set(te[:, 2])) - train_ents),
    }
    train_pairs = set(zip(tr[:, 0].tolist(), tr[:, 2].tolist()))
    for name, a in (("val", va), ("test", te)):
        res[f"{name}_pair_seen_in_train_other_relation"] = sum((h, t) in train_pairs for h, _, t in a.tolist())
        res[f"{name}_reversed_pair_in_train"] = sum((t, h) in train_pairs for h, _, t in a.tolist())
    return res


# ----------------------------------------------------------------------------- negative sampler
class UniformNegativeSampler:
    """Uniform corruption of head or tail. Known true triples (pass ALL splits) are filtered out.
    type_constrained=True draws the replacement only from entities of the same primary type."""

    def __init__(self, known_triples, num_entities, num_relations, entity_types=None,
                 type_constrained=False, seed=0, max_tries=20):
        self.N, self.R = num_entities, num_relations
        self.rng = np.random.default_rng(seed)
        self.max_tries = max_tries
        self.known = np.sort(self._key(known_triples))
        self.type_constrained = type_constrained
        if type_constrained:
            assert entity_types is not None
            self.ptype = entity_types.argmax(1)
            self.pools = {t: np.where(self.ptype == t)[0] for t in np.unique(self.ptype)}

    def _key(self, tri):
        tri = np.asarray(tri, dtype=np.int64)
        return (tri[..., 0] * self.R + tri[..., 1]) * self.N + tri[..., 2]

    def _draw(self, orig):
        if not self.type_constrained:
            return self.rng.integers(0, self.N, size=orig.shape)
        new = np.empty_like(orig)
        for t, pool in self.pools.items():
            m = self.ptype[orig] == t
            new[m] = pool[self.rng.integers(0, len(pool), size=m.sum())]
        return new

    def sample(self, pos, k=1, mode="both"):
        """pos: [B,3] -> negatives [B,k,3]. mode: 'both' (coin flip), 'head' or 'tail'."""
        pos = np.asarray(pos, dtype=np.int64)
        neg = np.repeat(pos[:, None, :], k, axis=1).copy()
        if mode == "both":
            col = np.where(self.rng.random(neg.shape[:2]) < 0.5, 0, 2)
        else:
            col = np.full(neg.shape[:2], 0 if mode == "head" else 2)
        bad = np.ones(neg.shape[:2], bool)
        for _ in range(self.max_tries):
            if not bad.any():
                break
            b, j = np.where(bad)
            orig = pos[b, col[b, j]]
            neg[b, j, col[b, j]] = self._draw(orig)
            changed = neg[b, j, col[b, j]] != orig
            is_true = np.isin(self._key(neg[b, j]), self.known)
            bad[b, j] = ~changed | is_true
        self.last_unresolved = int(bad.sum())  # slots still colliding after max_tries (normally 0)
        return neg


# ----------------------------------------------------------------------------- report
def write_report(path, args, raw, df, near, conflicts, ents, rels, tri, split, stats, types, leak):
    N = len(ents)
    deg = np.bincount(tri[:, 0], minlength=N) + np.bincount(tri[:, 2], minlength=N)
    L = []
    L.append("# CropDP-KG data report\n")
    L.append(f"Seed {args.seed}; val {args.val_frac}, test {args.test_frac}; near-duplicate merge: {args.merge_near_dups}; dropped relations: {args.drop_relations or 'none'}\n")
    L.append("## Cleaning\n")
    L.append(f"- raw rows: {len(raw)}")
    for s, c in df.status.value_counts().items():
        L.append(f"- {s}: {c}")
    L.append(f"- entities: {N}, relations: {len(rels)}, triples: {len(tri)}\n")
    L.append("## Per relation\n")
    L.append("| relation | triples | eligible for val/test | train | val | test |")
    L.append("|---|---|---|---|---|---|")
    for r, name in enumerate(rels):
        s = stats[r]
        m = tri[:, 1] == r
        L.append(f"| {name} | {s['total']} | {s['eligible']} | {(split[m] == 0).sum()} | "
                 f"{(split[m] == 1).sum()} | {(split[m] == 2).sum()} |")
    L.append("\n## Entity types (multi-hot counts)\n")
    for i, t in enumerate(TYPES):
        L.append(f"- {t}: {int(types[:, i].sum())}")
    L.append("\n## Graph shape\n")
    L.append(f"- mean degree {deg.mean():.2f}, median {np.median(deg):.0f}, max {deg.max()}")
    L.append(f"- entities with degree 1: {(deg == 1).sum()} ({(deg == 1).mean():.1%})")
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        trn = tri[split == 0]
        adj = coo_matrix((np.ones(len(trn)), (trn[:, 0], trn[:, 2])), shape=(N, N))
        nc, lab = connected_components(adj, directed=False)
        sizes = np.bincount(lab)
        L.append(f"- connected components (train graph): {nc}; largest has {sizes.max()} entities "
                 f"({sizes.max() / N:.1%}); components with <=3 entities: {(sizes <= 3).sum()}")
    except ImportError:
        L.append("- connected components: install scipy to compute")
    L.append("\n## Conflicts (kept and flagged, review by hand)\n")
    if len(conflicts):
        for k, c in conflicts.kind.value_counts().items():
            L.append(f"- {k}: {c}")
    else:
        L.append("- none")
    L.append("\n## Leakage checks (overlaps must be 0)\n")
    for k, v in leak.items():
        L.append(f"- {k}: {v}")
    sus = []
    for r, name in enumerate(rels):
        m = tri[:, 1] == r
        if name != "EnglishName" and m.sum() >= 100 and len(np.unique(tri[m, 0])) == m.sum():
            sus.append(name)
    numeric_heads = sum(bool(re.fullmatch(r"\d+", ents[i])) for i in set(tri[:, 0].tolist()))
    warn = [rels[r] for r, s in stats.items() if s["eligible"] < 0.5 * (s["val_target"] + s["test_target"])]
    L.append("\n## Warnings\n")
    if numeric_heads:
        L.append(f"- SUSPICIOUS HEADS: {numeric_heads} head entities are pure numbers; the first column is probably "
                 f"a row index, not a pest/disease name. Run --inspect.")
    elif sus:
        L.append(f"- note: relations with exactly one triple per head: {sus} (fine if each pest has a single value).")
    if warn:
        L.append("- Too few eligible triples to reach the val/test targets for: " + ", ".join(warn)
                 + ". Most of their tails occur only once, so they cannot be held out without leaving "
                   "entities unseen. Consider excluding these relations from the headline metrics.")
    else:
        L.append("- none")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- demo data
def write_demo(path):
    rng = np.random.default_rng(0)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    crops = [f"作物{i}" for i in range(25)]
    areas = [f"省{i}" for i in range(15)]
    parts = ["叶片", "根", "果实", "茎", "花"]
    syms = [f"症状{i}" for i in range(40)]
    conds = [f"条件{i}" for i in range(30)]
    spec = {"relationEng.csv": ("EnglishName", None), "relationarea.csv": ("RegionIs", areas),
            "relationcon.csv": ("ConditionIs", conds), "relationcrop.csv": ("Damage", crops),
            "relationpart.csv": ("InflictHarmPart", parts), "relationsym.csv": ("ManifestAs", syms),
            "relationtem.csv": ("TemperIs", [f"{a}-{a + 5}℃" for a in range(10, 30, 2)])}
    for fn, (rel, pool) in spec.items():
        rows = []
        for i in range(300):
            head = f"病虫害{i}"
            if rel == "EnglishName":
                rows.append((head, rel, f"Pest Name {i}"))
            else:
                for t in rng.choice(pool, size=rng.integers(1, 4), replace=False):
                    rows.append((head, rel, t))
        rows += [rows[0], ("病虫害0 ", rel, rows[0][2]), ("", rel, "x")]  # duplicate, whitespace variant, missing
        pd.DataFrame(rows, columns=["head", "relation", "tail"]).to_csv(path / fn, index=False, encoding="utf-8-sig")
    print(f"Demo CSVs written to {path}")


# ----------------------------------------------------------------------------- main
def save_arr(out, name, arr):
    np.save(out / f"{name}.npy", arr)
    if torch is not None:
        torch.save(torch.from_numpy(np.ascontiguousarray(arr)), out / f"{name}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--merge-near-dups", action="store_true", help="merge entities that differ only by case/spacing")
    ap.add_argument("--drop-relations", default="", help="comma list, e.g. EnglishName (entities left with no triples vanish)")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="print the head of each CSV and exit")
    ap.add_argument("--demo", metavar="DIR", help="write a tiny synthetic dataset to DIR and exit")
    args = ap.parse_args()

    if args.demo:
        return write_demo(args.demo)
    if args.inspect:
        return inspect(args.raw, not args.no_header)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.raw, not args.no_header)
    df, near = clean(raw, args.merge_near_dups)
    kept = df[df.status == "kept"].copy()
    args.drop_relations = [x.strip() for x in args.drop_relations.split(",") if x.strip()]
    if args.drop_relations:
        mask = kept.r.isin(args.drop_relations)
        print(f"[drop] removing {int(mask.sum())} triples of relations {args.drop_relations}")
        df.loc[kept.index[mask], "status"] = "dropped_by_flag"
        kept = kept[~mask]
    conflicts = find_conflicts(kept, near, args.merge_near_dups)
    ents, rels, e2i, r2i, tri, types = build_ids(kept)
    R, N = len(rels), len(ents)

    split, stats = make_splits(tri, N, args.val_frac, args.test_frac, args.seed)
    # a val/test triple whose (head, tail) pair already occurs in train under another relation is a leak
    train_pairs = set(zip(tri[split == 0][:, 0].tolist(), tri[split == 0][:, 2].tolist()))
    moved = [i for i in np.where(split > 0)[0] if (int(tri[i, 0]), int(tri[i, 2])) in train_pairs]
    split[moved] = 0
    leak = leakage_check(tri, split, N, R)
    leak["moved_to_train_for_pair_leak"] = len(moved)
    assert leak["train_val_overlap"] == leak["train_test_overlap"] == leak["val_test_overlap"] == 0, leak
    assert leak["val_entities_missing_in_train"] == leak["test_entities_missing_in_train"] == 0, leak
    tr, va, te = tri[split == 0], tri[split == 1], tri[split == 2]

    # provenance: every raw row, what happened to it, and the final ids
    prov = df[["file", "line", "raw_h", "raw_r", "raw_t", "h", "r", "t", "status"]].rename(
        columns={"h": "norm_h", "r": "norm_r", "t": "norm_t"})
    prov.to_csv(out / "provenance.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(out / "conflicts.csv", index=False, encoding="utf-8-sig")
    en_map = english_map(df)
    pd.DataFrame({"id": range(N), "name": ents, "english": [en_map.get(e, "") for e in ents], "types": [
        "|".join(t for t, v in zip(TYPES, row) if v) for row in types]}).to_csv(
        out / "entities.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": range(R), "name": rels}).to_csv(out / "relations.csv", index=False)
    pd.DataFrame({"head": [ents[i] for i in tri[:, 0]], "relation": [rels[i] for i in tri[:, 1]],
                  "tail": [ents[i] for i in tri[:, 2]],
                  "split": np.array(["train", "val", "test"])[split]}).to_csv(
        out / "triples_clean.csv", index=False, encoding="utf-8-sig")

    for name, a in (("train", tr), ("val", va), ("test", te)):
        save_arr(out, name, a)
    save_arr(out, "node_types", types)

    # train-only message-passing graph with inverse edges (relation id r + R = inverse of r)
    edge_index = np.concatenate([tr[:, [0, 2]].T, tr[:, [2, 0]].T], 1).astype(np.int64)
    edge_type = np.concatenate([tr[:, 1], tr[:, 1] + R]).astype(np.int64)
    save_arr(out, "graph_edge_index", edge_index)
    save_arr(out, "graph_edge_type", edge_type)

    # fixed negatives for AUC: 1 per positive, filtered against all known triples
    sampler = UniformNegativeSampler(tri, N, R, seed=args.seed)
    save_arr(out, "val_neg", sampler.sample(va, 1)[:, 0, :])
    save_arr(out, "test_neg", sampler.sample(te, 1)[:, 0, :])

    write_report(out / "data_report.md", args, raw, df, near, conflicts, ents, rels, tri, split, stats, types, leak)
    meta = dict(seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac, num_entities=N,
                num_relations=R, num_relations_with_inverse=2 * R, types=TYPES, relations=rels,
                n_train=len(tr), n_val=len(va), n_test=len(te), merge_near_dups=args.merge_near_dups)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"entities {N} | relations {R} | train {len(tr)} val {len(va)} test {len(te)} | "
          f"conflicts flagged {len(conflicts)}")
    print(f"Outputs in {out}. Read data_report.md first.")


if __name__ == "__main__":
    main()
