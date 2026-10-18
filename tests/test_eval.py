"""Owner: Person 4. Unit tests for benchmark evaluator (filtered ranking and AUC)."""
import pytest
import torch
import numpy as np

from src.eval.evaluator import evaluate_ranking, evaluate_auc, build_known_tails_dict


def test_build_known_tails_dict():
    triples = torch.tensor([
        [0, 1, 10],
        [0, 1, 20],
        [1, 2, 30],
    ])
    d = build_known_tails_dict(triples)
    assert d[(0, 1)] == {10, 20}
    assert d[(1, 2)] == {30}
    assert d[(0, 2)] == set()


def test_evaluate_ranking_filtered_correctness():
    num_entities = 10
    # Queries:
    # Query 0: (0, 0, 3)
    # Query 1: (1, 0, 5)
    triples = torch.tensor([[0, 0, 3], [1, 0, 5]])
    # Another known true tail for (0, 0) is 2, which has a higher raw score than target 3
    known_triples = torch.tensor([[0, 0, 3], [0, 0, 2], [1, 0, 5]])

    def mock_score_all_tails(h, r):
        # Return synthetic scores: entity 2 scores 10.0, entity 3 scores 5.0, others 0.0
        scores = torch.zeros((len(h), num_entities))
        scores[0, 2] = 10.0
        scores[0, 3] = 5.0
        scores[0, 4] = 7.0  # An unknown tail with higher score -> should beat target
        scores[1, 5] = 9.0  # Top score for query 1
        return scores

    res = evaluate_ranking(
        score_all_tails=mock_score_all_tails,
        triples=triples,
        known_triples=known_triples,
        num_entities=num_entities,
    )

    # For query 0:
    # Entity 2 is another known tail -> filtered to -1e9.
    # Entity 4 scores 7.0 > 5.0 -> beats target.
    # So target 3 ranks at position 2!
    # For query 1: target 5 scores 9.0 (highest) -> rank 1.
    # MRR = (1/2 + 1/1) / 2 = 0.75
    # Hits@1 = 0.5
    # Hits@3 = 1.0
    assert pytest.approx(res["mrr"], abs=1e-4) == 0.75
    assert pytest.approx(res["hits@1"], abs=1e-4) == 0.5
    assert pytest.approx(res["hits@3"], abs=1e-4) == 1.0


def test_evaluate_auc_perfect():
    pos = torch.tensor([[0, 0, 1], [1, 0, 2]])
    neg = torch.tensor([[0, 0, 3], [1, 0, 4]])

    def mock_score_fn(triples):
        # Positives get 1.0, negatives get -1.0
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        return torch.where(t <= 2, 1.0, -1.0)

    auc = evaluate_auc(mock_score_fn, pos, neg)
    assert pytest.approx(auc, abs=1e-4) == 1.0
