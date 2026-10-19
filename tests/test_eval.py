"""Owner: Person 4. Unit tests for the unified evaluator (filtered ranking, average ties, AUC)."""
import pytest
import torch

from src.eval.evaluator import evaluate_ranking, evaluate_auc, build_known_tails_dict


# ---- Person 2's original tests (kept) -------------------------------------------------
def test_build_known_tails_dict():
    triples = torch.tensor([[0, 1, 10], [0, 1, 20], [1, 2, 30]])
    d = build_known_tails_dict(triples)
    assert d[(0, 1)] == {10, 20}
    assert d[(1, 2)] == {30}
    assert d[(0, 2)] == set()


def test_evaluate_ranking_filtered_correctness():
    num_entities = 10
    triples = torch.tensor([[0, 0, 3], [1, 0, 5]])
    known_triples = torch.tensor([[0, 0, 3], [0, 0, 2], [1, 0, 5]])

    def mock_score_all_tails(h, r):
        scores = torch.zeros((len(h), num_entities))
        scores[0, 2] = 10.0
        scores[0, 3] = 5.0
        scores[0, 4] = 7.0
        scores[1, 5] = 9.0
        return scores

    res = evaluate_ranking(mock_score_all_tails, triples, known_triples, num_entities)
    assert pytest.approx(res["mrr"], abs=1e-4) == 0.75
    assert pytest.approx(res["hits@1"], abs=1e-4) == 0.5
    assert pytest.approx(res["hits@3"], abs=1e-4) == 1.0


def test_evaluate_auc_perfect():
    pos = torch.tensor([[0, 0, 1], [1, 0, 2]])
    neg = torch.tensor([[0, 0, 3], [1, 0, 4]])

    def mock_score_fn(triples):
        t = triples[:, 2]
        return torch.where(t <= 2, 1.0, -1.0)

    assert pytest.approx(evaluate_auc(mock_score_fn, pos, neg), abs=1e-4) == 1.0


# ---- Person 4's tests -----------------------------------------------------------------
N = 10
known = torch.tensor([[0, 0, 1], [0, 0, 2], [3, 1, 4]])


def perfect(h, r):
    s = torch.zeros(len(h), N)
    for i, (hh, rr) in enumerate(zip(h.tolist(), r.tolist())):
        for t in known[(known[:, 0] == hh) & (known[:, 1] == rr)][:, 2].tolist():
            s[i, t] = 1.0
    return s


def test_perfect_gives_mrr_1_thanks_to_filtering():
    out = evaluate_ranking(perfect, known, known, N)
    assert out["MRR"] == pytest.approx(1.0)


def test_all_equal_scores_gives_average_rank():
    const = lambda h, r: torch.zeros(len(h), N)
    out = evaluate_ranking(const, known[2:3], known, N)
    assert out["MRR"] == pytest.approx(1 / 5.5)


def test_per_relation_keys():
    out = evaluate_ranking(perfect, known, known, N)
    assert "r0/MRR" in out and "r1/MRR" in out and "macro/MRR" in out


def test_auc_perfect_and_constant():
    pos, neg = torch.zeros(5, 3).long(), torch.ones(5, 3).long()
    assert evaluate_auc(lambda x: x[:, 0].float(), neg, pos) == 1.0
    assert evaluate_auc(lambda x: torch.zeros(len(x)), pos, neg) == 0.5


def test_backward_compatible_keys_and_average_ties():
    const = lambda h, r: torch.zeros(len(h), N)
    out = evaluate_ranking(const, known[2:3], known, N, relation_names=["a", "b"])
    assert out["mrr"] == pytest.approx(out["MRR"])
    assert out["per_relation"][1]["name"] == "b" and out["per_relation"][1]["count"] == 1
    assert out["mean_rank"] == pytest.approx(5.5)