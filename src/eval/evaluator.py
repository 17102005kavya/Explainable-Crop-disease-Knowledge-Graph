"""Owner: Person 4. THE ONE evaluator used by every model."""
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ScoreAllTails = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _metrics(ranks: torch.Tensor) -> Dict[str, float]:
    return {
        "MRR": (1.0 / ranks).mean().item(),
        "Hits@1": (ranks <= 1).float().mean().item(),
        "Hits@3": (ranks <= 3).float().mean().item(),
        "Hits@10": (ranks <= 10).float().mean().item(),
    }


@torch.no_grad()
def evaluate_ranking(score_all_tails: ScoreAllTails, triples: torch.Tensor,
                     known_triples: torch.Tensor, num_entities: int,
                     batch_size: int = 256) -> Dict[str, float]:
    """Filtered tail prediction. Returns a flat dict:
    MRR, Hits@1/3/10, n            -> micro average over all triples
    macro/MRR, ...                 -> mean over relations (not dominated by ManifestAs)
    r{ID}/MRR, ..., r{ID}/n        -> per relation
    """
    known = defaultdict(set)
    for h, r, t in known_triples.tolist():
        known[(h, r)].add(t)

    all_ranks, all_rels = [], []
    for s in range(0, len(triples), batch_size):
        batch = triples[s:s + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        scores = score_all_tails(h, r).float().clone()               # [B, N]
        assert scores.shape == (len(batch), num_entities), scores.shape
        rows = torch.arange(len(batch), device=scores.device)
        true = scores[rows, t.to(scores.device)].clone()             # score of true tail
        for i, (hi, ri, ti) in enumerate(batch.tolist()):            # FILTER other true tails
            others = [x for x in known[(hi, ri)] if x != ti]
            if others:
                scores[i, others] = float("-inf")
        greater = (scores > true[:, None]).sum(1)
        ties = (scores == true[:, None]).sum(1) - 1                  # minus itself
        ranks = 1.0 + greater.float() + ties.float() / 2.0           # average rank for ties
        all_ranks.append(ranks.cpu())
        all_rels.append(r.cpu())

    ranks, rels = torch.cat(all_ranks), torch.cat(all_rels)
    out = _metrics(ranks)
    out["n"] = float(len(ranks))
    per_rel = {}
    for rid in rels.unique().tolist():
        m = _metrics(ranks[rels == rid])
        per_rel[rid] = m
        out.update({f"r{rid}/{k}": v for k, v in m.items()})
        out[f"r{rid}/n"] = float((rels == rid).sum())
    for k in ["MRR", "Hits@1", "Hits@3", "Hits@10"]:
        out[f"macro/{k}"] = float(np.mean([m[k] for m in per_rel.values()]))
    return out


@torch.no_grad()
def evaluate_auc(score_fn: Callable[[torch.Tensor], torch.Tensor],
                 pos: torch.Tensor, neg: torch.Tensor) -> float:
    """AUC of positives vs the fixed negatives. score_fn(triples[M,3]) -> [M]."""
    sp = score_fn(pos).float().cpu().numpy()
    sn = score_fn(neg).float().cpu().numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    return float(roc_auc_score(y, np.r_[sp, sn]))