import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.eval.evaluator import evaluate_ranking
from src.eval.vrgae_adapter import VRGAEScorer, fit_shared_recipe


def _tiny_data(N=30, n=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    tr = torch.stack([torch.randint(0, N, (n,), generator=g),
                      torch.randint(0, 6, (n,), generator=g),
                      torch.randint(0, N, (n,), generator=g)], 1)
    ei = torch.cat([torch.stack([tr[:, 0], tr[:, 2]]), torch.stack([tr[:, 2], tr[:, 0]])], 1)
    et = torch.cat([tr[:, 1], tr[:, 1] + 6])
    nt = torch.zeros(N, 8)
    nt[torch.arange(N), torch.randint(0, 7, (N,), generator=g)] = 1.0
    return dict(train=tr, val=tr[:10], test=tr[10:20], node_types=nt, edge_index=ei, edge_type=et,
                val_neg=tr[:10], test_neg=tr[10:20], meta={"num_entities": N, "num_relations": 6})


def test_vrgae_adapter_trains_and_evaluates():
    try:
        from src.model.vrgae import VRGAE
    except Exception as e:                                   # main still has stub latent.py / loss.py
        pytest.skip(f"VRGAE not importable yet: {e}")
    d = _tiny_data()
    cfg = {"encoder": {"emb_dim": 8, "hidden_dim": 16, "out_dim": 16, "num_relations": 12,
                       "num_bases": 2, "dropout": 0.1},
           "decoder": {"num_base_relations": 6, "gamma": 6.0},
           "model": {"decoder": "rotate"}}
    model = VRGAE(d["node_types"], cfg)
    ep, mrr = fit_shared_recipe(model, d, epochs=2, batch_size=16, num_neg=4, eval_every=1,
                                log=lambda *_: None)
    assert 1 <= ep <= 2 and 0.0 < mrr <= 1.0
    sc = VRGAEScorer(model, d["edge_index"], d["edge_type"])
    known = torch.cat([d["train"], d["val"], d["test"]])
    res = evaluate_ranking(sc.make_score_all_tails(), d["test"], known, 30)
    assert res["n"] == 10.0
    assert sc.score_triples(d["test"]).shape == (10,)