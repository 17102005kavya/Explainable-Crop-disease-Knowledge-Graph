"""Owner: Person 4 (Part D). Baselines.
Each exposes make_score_all_tails() -> score_all_tails(h, r) -> [B, N]
and score_triples(triples[M,3]) -> [M] (used for AUC)."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.evaluator import evaluate_ranking


class PopularityBaseline:
    """Sanity check: rank tails by how often they appear as the tail of relation r in train."""

    def __init__(self, train: torch.Tensor, num_entities: int, num_relations: int = 6):
        counts = torch.zeros(num_relations, num_entities)
        ones = torch.ones(len(train))
        counts.index_put_((train[:, 1], train[:, 2]), ones, accumulate=True)
        self.counts = counts                                   # [R, N]

    def make_score_all_tails(self):
        def score_all_tails(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return self.counts[r.cpu()]                        # [B, N], ignores h
        return score_all_tails

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return self.counts[triples[:, 1].cpu(), triples[:, 2].cpu()]


class RotatEBaseline(nn.Module):
    """RotatE alone: plain entity embeddings (complex), relations are rotations.
    score(h, r, t) = gamma - sum | h * e^(i*phase_r) - t |   (no graph encoder, no variational part)."""

    def __init__(self, num_entities: int, num_relations: int = 6, dim: int = 64, gamma: float = 6.0):
        super().__init__()
        self.N, self.dim, self.gamma = num_entities, dim, gamma
        self.ent = nn.Embedding(num_entities, 2 * dim)         # [re | im]  -> 128 real numbers for dim=64
        self.rel = nn.Embedding(num_relations, dim)            # rotation angles
        bound = (gamma + 2.0) / dim
        nn.init.uniform_(self.ent.weight, -bound, bound)
        nn.init.uniform_(self.rel.weight, -math.pi, math.pi)

    def _rotate(self, h_ids, r_ids):
        hre, him = self.ent(h_ids).chunk(2, dim=-1)
        ph = self.rel(r_ids)
        cos, sin = torch.cos(ph), torch.sin(ph)
        return hre * cos - him * sin, hre * sin + him * cos

    def score(self, h, r, t):
        rre, rim = self._rotate(h, r)
        tre, tim = self.ent(t).chunk(2, dim=-1)
        dist = torch.sqrt((rre - tre) ** 2 + (rim - tim) ** 2 + 1e-9).sum(-1)
        return self.gamma - dist                               # [B]

    def score_all_tails(self, h, r):
        rre, rim = self._rotate(h, r)                          # [B, dim]
        all_re, all_im = self.ent.weight.chunk(2, dim=-1)      # [N, dim]
        out = []
        for s in range(0, self.N, 2048):                       # chunk entities to save memory
            ere, eim = all_re[s:s + 2048], all_im[s:s + 2048]
            d = torch.sqrt((rre[:, None, :] - ere[None]) ** 2 +
                           (rim[:, None, :] - eim[None]) ** 2 + 1e-9).sum(-1)
            out.append(self.gamma - d)
        return torch.cat(out, dim=1)                           # [B, N]

    # ---- shared-scorer interface -------------------------------------------------
    def make_score_all_tails(self):
        return lambda h, r: self.score_all_tails(h, r)

    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        return self.score(triples[:, 0], triples[:, 1], triples[:, 2])

    # ---- training ---------------------------------------------------------------
    def fit(self, data: dict, epochs=200, lr=1e-3, num_neg=16, batch_size=1024,
            adv_temp=1.0, eval_every=10, seed=42, log=print):
        """Self-adversarial negative sampling. Picks the epoch with the best VAL MRR."""
        torch.manual_seed(seed)
        train, val = data["train"], data["val"]
        known = torch.cat([data["train"], data["val"], data["test"]])
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        best_mrr, best_state, best_ep = -1.0, None, 0

        for ep in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(len(train))
            total = 0.0
            for s in range(0, len(train), batch_size):
                b = train[perm[s:s + batch_size]]
                h, r, t = b[:, 0], b[:, 1], b[:, 2]
                B = len(b)
                pos = self.score(h, r, t)                                          # [B]
                rand = torch.randint(0, self.N, (B, num_neg))
                head_side = torch.rand(B, 1) < 0.5                                 # corrupt head or tail
                hn = torch.where(head_side, rand, h[:, None].expand(B, num_neg))
                tn = torch.where(head_side, t[:, None].expand(B, num_neg), rand)
                rn = r[:, None].expand(B, num_neg)
                neg = self.score(hn.reshape(-1), rn.reshape(-1), tn.reshape(-1)).view(B, num_neg)
                w = torch.softmax(adv_temp * neg, dim=1).detach()
                loss = -F.logsigmoid(pos).mean() - (w * F.logsigmoid(-neg)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * B
            if ep % eval_every == 0 or ep == epochs:
                self.eval()
                res = evaluate_ranking(self.make_score_all_tails(), val, known, self.N)
                log(f"epoch {ep:4d} | loss {total / len(train):.4f} | val MRR {res['MRR']:.4f} "
                    f"H@10 {res['Hits@10']:.4f}")
                if res["MRR"] > best_mrr:
                    best_mrr, best_ep, best_state = res["MRR"], ep, copy.deepcopy(self.state_dict())
        self.load_state_dict(best_state)
        self.eval()
        log(f"best epoch {best_ep} (val MRR {best_mrr:.4f})")
        return best_ep, best_mrr


class RGCNDistMultBaseline:
    """R-GCN encoder + DistMult decoder, no variational part (Step 4)."""