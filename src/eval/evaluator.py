"""Owner: Person 4. THE ONE evaluator used by every model (filtered tail prediction + AUC).

Returns BOTH key styles so every script keeps working:
  MRR, Hits@1/3/10, n, macro/..., r{ID}/...        (Person 4 style, flat)
  mrr, hits@1/3/10, mean_rank, num_eval, per_relation (Person 2 style)
Ties get the average rank, so models that output equal scores are not rewarded.
"""
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ScoreAllTails = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (h[B], r[B]) -> [B, N]


def build_known_tails_dict(known_triples) -> Dict[tuple, set]:
    """Map (head, relation) -> set of true tails for filtered evaluation."""
    known_np = known_triples.cpu().numpy() if isinstance(known_triples, torch.Tensor) else np.asarray(known_triples)
    known = defaultdict(set)
    for h, r, t in known_np:
        known[(int(h), int(r))].add(int(t))
    return known


def _summary(ranks: torch.Tensor) -> Dict[str, float]:
    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits@1": (ranks <= 1).float().mean().item(),
        "hits@3": (ranks <= 3).float().mean().item(),
        "hits@10": (ranks <= 10).float().mean().item(),
        "mean_rank": ranks.mean().item(),
    }


@torch.no_grad()
def evaluate_ranking(score_all_tails: ScoreAllTails, triples: torch.Tensor,
                     known_triples, num_entities: Optional[int] = None,
                     batch_size: int = 256,
                     relation_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Filtered tail prediction over all N candidates.
    known_triples = train + val + test as a [M,3] tensor, OR a ready-made {(h, r): {tails}} dict.
    num_entities is optional (only used as a shape check)."""
    known = known_triples if isinstance(known_triples, dict) else build_known_tails_dict(known_triples)

    all_ranks, all_rels = [], []
    for s in range(0, len(triples), batch_size):
        batch = triples[s:s + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
        scores = score_all_tails(h, r).float().clone()                # [B, N]
        if num_entities is not None:
            assert scores.shape == (len(batch), num_entities), scores.shape
        rows = torch.arange(len(batch), device=scores.device)
        true = scores[rows, t.to(scores.device)].clone()              # score of the true tail
        for i, (hi, ri, ti) in enumerate(batch.tolist()):             # filter OTHER true tails
            others = [x for x in known.get((hi, ri), ()) if x != ti]
            if others:
                scores[i, others] = float("-inf")
        greater = (scores > true[:, None]).sum(1)
        ties = (scores == true[:, None]).sum(1) - 1                   # minus the true tail itself
        ranks = 1.0 + greater.float() + ties.float() / 2.0            # average rank for ties
        all_ranks.append(ranks.cpu())
        all_rels.append(r.cpu())

    ranks, rels = torch.cat(all_ranks), torch.cat(all_rels)
    s = _summary(ranks)
    out: Dict[str, Any] = {
        "MRR": s["mrr"], "Hits@1": s["hits@1"], "Hits@3": s["hits@3"], "Hits@10": s["hits@10"],
        "n": float(len(ranks)),
        "mrr": s["mrr"], "hits@1": s["hits@1"], "hits@3": s["hits@3"], "hits@10": s["hits@10"],
        "mean_rank": s["mean_rank"], "num_eval": len(ranks),
    }
    per_relation = {}
    for rid in rels.unique().tolist():
        m = rels == rid
        sr = _summary(ranks[m])
        cnt = int(m.sum())
        name = relation_names[rid] if relation_names and rid < len(relation_names) else str(rid)
        per_relation[rid] = {"name": name, "count": cnt, **sr}
        out[f"r{rid}/MRR"], out[f"r{rid}/Hits@1"] = sr["mrr"], sr["hits@1"]
        out[f"r{rid}/Hits@3"], out[f"r{rid}/Hits@10"] = sr["hits@3"], sr["hits@10"]
        out[f"r{rid}/n"] = float(cnt)
    out["per_relation"] = per_relation
    for k, low in [("MRR", "mrr"), ("Hits@1", "hits@1"), ("Hits@3", "hits@3"), ("Hits@10", "hits@10")]:
        out[f"macro/{k}"] = float(np.mean([v[low] for v in per_relation.values()]))
    return out


@torch.no_grad()
def evaluate_auc(score_fn: Callable[[torch.Tensor], torch.Tensor],
                 pos: torch.Tensor, neg: torch.Tensor) -> float:
    """ROC-AUC of positives vs the fixed negatives (ties handled). score_fn(triples[M,3]) -> [M]."""
    sp = score_fn(pos).float().reshape(-1).cpu().numpy()
    sn = score_fn(neg).float().reshape(-1).cpu().numpy()
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    return float(roc_auc_score(y, np.r_[sp, sn]))