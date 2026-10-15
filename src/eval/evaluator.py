"""Owner: Person 4 (Part D). THE ONE evaluator used by every model.  TODO: implement."""
from typing import Callable, Dict

import torch

ScoreAllTails = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]   # (h[B], r[B]) -> [B, N]


def evaluate_ranking(score_all_tails: ScoreAllTails, triples: torch.Tensor, known_triples: torch.Tensor,
                     num_entities: int, batch_size: int = 256) -> Dict[str, float]:
    """Filtered tail-prediction ranking.

    For each (h, r, t): score all N tails, set the scores of OTHER known true tails of (h, r) (from known_triples =
    train + val + test) to -inf, rank the true tail. Return overall and per-relation
    {"MRR", "Hits@1", "Hits@3", "Hits@10"}.
    """
    raise NotImplementedError


def evaluate_auc(score_fn: Callable[[torch.Tensor], torch.Tensor], pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUC of positive vs the fixed negatives (val_neg / test_neg).  score_fn(triples[M,3]) -> [M]."""
    raise NotImplementedError
