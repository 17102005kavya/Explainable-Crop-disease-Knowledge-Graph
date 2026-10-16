import pytest
torch = pytest.importorskip("torch")
from src.eval.evaluator import evaluate_ranking, evaluate_auc

N = 10
known = torch.tensor([[0, 0, 1], [0, 0, 2], [3, 1, 4]])


def perfect(h, r):
    """Gives score 1 to every true tail of (h, r), 0 to everything else."""
    s = torch.zeros(len(h), N)
    for i, (hh, rr) in enumerate(zip(h.tolist(), r.tolist())):
        for t in known[(known[:, 0] == hh) & (known[:, 1] == rr)][:, 2].tolist():
            s[i, t] = 1.0
    return s


def test_perfect_gives_mrr_1_thanks_to_filtering():
    # (0,0,1) and (0,0,2) tie at score 1; filtering removes the other true tail -> rank 1
    out = evaluate_ranking(perfect, known, known, N)
    assert out["MRR"] == pytest.approx(1.0)


def test_all_equal_scores_gives_average_rank():
    const = lambda h, r: torch.zeros(len(h), N)
    out = evaluate_ranking(const, known[2:3], known, N)    # 10 tied, nothing to filter
    assert out["MRR"] == pytest.approx(1 / 5.5)


def test_per_relation_keys():
    out = evaluate_ranking(perfect, known, known, N)
    assert "r0/MRR" in out and "r1/MRR" in out and "macro/MRR" in out


def test_auc_perfect_and_constant():
    pos, neg = torch.zeros(5, 3).long(), torch.ones(5, 3).long()
    assert evaluate_auc(lambda x: x[:, 0].float(), neg, pos) == 1.0
    assert evaluate_auc(lambda x: torch.zeros(len(x)), pos, neg) == 0.5