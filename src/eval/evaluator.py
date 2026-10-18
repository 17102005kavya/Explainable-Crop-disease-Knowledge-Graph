"""Owner: Person 4 (Part D). Unified benchmark evaluator for all models.

Implements filtered tail-prediction ranking across all candidate entities,
per-relation metric breakdowns, and ROC-AUC evaluation.
"""
from collections import defaultdict
from typing import Callable, Dict, Any, List, Optional
import numpy as np
import torch

ScoreAllTails = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (h[B], r[B]) -> [B, N]


def build_known_tails_dict(known_triples: torch.Tensor) -> Dict[tuple, set]:
    """Map (head, relation) -> set of true tails for filtered evaluation."""
    known_np = known_triples.cpu().numpy() if isinstance(known_triples, torch.Tensor) else np.asarray(known_triples)
    known_tails = defaultdict(set)
    for h, r, t in known_np:
        known_tails[(int(h), int(r))].add(int(t))
    return known_tails


@torch.no_grad()
def evaluate_ranking(
    score_all_tails: ScoreAllTails,
    triples: torch.Tensor,
    known_triples: torch.Tensor,
    num_entities: int,
    batch_size: int = 256,
    relation_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Filtered tail-prediction ranking evaluation.

    For each (h, r, t):
    1. Score all N candidate entities using score_all_tails(h, r).
    2. Filter out other known true tails of (h, r) from known_triples = train + val + test
       by setting their score to -1e9, strictly preserving the target tail.
    3. Rank target tail among candidates: rank = 1 + sum(score > target_score).
    4. Compute overall and per-relation MRR, Hits@1, Hits@3, and Hits@10.
    """
    known_tails = build_known_tails_dict(known_triples)

    ranks = []
    ranks_by_rel = defaultdict(list)

    num_eval = len(triples)
    for i in range(0, num_eval, batch_size):
        batch = triples[i:i + batch_size]
        h = batch[:, 0]
        r = batch[:, 1]
        t = batch[:, 2]

        scores = score_all_tails(h, r)  # [B, N]

        for j in range(len(batch)):
            hj = int(h[j])
            rj = int(r[j])
            tj = int(t[j])
            target_score = scores[j, tj].item()

            row_scores = scores[j].clone()
            filter_tails = known_tails.get((hj, rj), set())
            for ft in filter_tails:
                if ft != tj:
                    row_scores[ft] = -1e9

            rank = (row_scores > target_score).sum().item() + 1
            ranks.append(rank)
            ranks_by_rel[rj].append(rank)

    ranks_arr = np.array(ranks, dtype=np.float64)
    overall = {
        "mrr": float(np.mean(1.0 / ranks_arr)),
        "hits@1": float(np.mean(ranks_arr <= 1.0)),
        "hits@3": float(np.mean(ranks_arr <= 3.0)),
        "hits@10": float(np.mean(ranks_arr <= 10.0)),
        "mean_rank": float(np.mean(ranks_arr)),
        "num_eval": len(ranks),
    }

    per_relation = {}
    for rj, r_ranks in ranks_by_rel.items():
        arr = np.array(r_ranks, dtype=np.float64)
        name = relation_names[rj] if relation_names and rj < len(relation_names) else str(rj)
        per_relation[rj] = {
            "name": name,
            "count": len(arr),
            "mrr": float(np.mean(1.0 / arr)),
            "hits@1": float(np.mean(arr <= 1.0)),
            "hits@3": float(np.mean(arr <= 3.0)),
            "hits@10": float(np.mean(arr <= 10.0)),
            "mean_rank": float(np.mean(arr)),
        }

    overall["per_relation"] = per_relation
    return overall


def evaluate_auc(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
    neg: torch.Tensor,
) -> float:
    """Compute ROC-AUC of positive vs fixed negative triples."""
    with torch.no_grad():
        pos_scores = score_fn(pos).cpu().numpy().reshape(-1)
        neg_scores = score_fn(neg).cpu().numpy().reshape(-1)

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_scores = np.concatenate([pos_scores, neg_scores])

    # Rank-sum calculation of AUC (Mann-Whitney U)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    order = np.argsort(y_scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_scores)) + 1
    u = np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))
