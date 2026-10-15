"""Owner: Person 1. Sanity checks on data/processed (numpy only). Skipped if the data is not there."""
import json
from pathlib import Path

import numpy as np
import pytest

D = Path("data/processed")
pytestmark = pytest.mark.skipif(not (D / "meta.json").exists(), reason="data/processed not found")


def L(name):
    return np.load(D / f"{name}.npy")


def keyset(a):
    return set(map(tuple, a.tolist()))


def test_shapes_and_ranges():
    meta = json.loads((D / "meta.json").read_text(encoding="utf-8"))
    N, R = meta["num_entities"], meta["num_relations"]
    for n in ("train", "val", "test", "val_neg", "test_neg"):
        a = L(n)
        assert a.ndim == 2 and a.shape[1] == 3
        assert a[:, [0, 2]].max() < N and a[:, 1].max() < R
    assert L("node_types").shape[0] == N


def test_splits_do_not_overlap():
    tr, va, te = keyset(L("train")), keyset(L("val")), keyset(L("test"))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_eval_entities_seen_in_train():
    tr = L("train")
    seen = set(tr[:, 0].tolist()) | set(tr[:, 2].tolist())
    for n in ("val", "test"):
        a = L(n)
        assert (set(a[:, 0].tolist()) | set(a[:, 2].tolist())) <= seen


def test_graph_is_train_only_with_inverse_edges():
    tr, ei, et = L("train"), L("graph_edge_index"), L("graph_edge_type")
    R = json.loads((D / "meta.json").read_text(encoding="utf-8"))["num_relations"]
    n = len(tr)
    assert ei.shape == (2, 2 * n) and len(et) == 2 * n
    assert np.array_equal(ei[:, :n], tr[:, [0, 2]].T) and np.array_equal(et[:n], tr[:, 1])
    assert np.array_equal(ei[:, n:], tr[:, [2, 0]].T) and np.array_equal(et[n:], tr[:, 1] + R)


def test_fixed_negatives_are_not_true_triples():
    known = keyset(np.concatenate([L("train"), L("val"), L("test")]))
    for n in ("val_neg", "test_neg"):
        assert not (keyset(L(n)) & known)
