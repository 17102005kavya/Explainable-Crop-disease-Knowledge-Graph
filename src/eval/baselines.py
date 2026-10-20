"""Owner: Person 4 (Part D). Baselines.
Each exposes make_score_all_tails() -> score_all_tails(h, r) -> [B, N]
and score_triples(triples[M,3]) -> [M] (used for AUC)."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.prepare_cropdp import UniformNegativeSampler      # the team's ONE negative sampler
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
        sampler = UniformNegativeSampler(known.numpy(), self.N, data["meta"]["num_relations"], seed=seed)
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
                negs = torch.from_numpy(sampler.sample(b.numpy(), k=num_neg)).long()   # shared sampler -> [B, K, 3]
                hn, rn, tn = negs[..., 0], negs[..., 1], negs[..., 2]
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


class RGCNLayer(nn.Module):
    """Relational GCN layer with basis decomposition and per-(node, relation) mean normalisation."""

    def __init__(self, in_dim, out_dim, num_rel, num_bases):
        super().__init__()
        self.num_rel = num_rel
        self.bases = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
        self.coef = nn.Parameter(torch.empty(num_rel, num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim)
        for b in range(num_bases):
            nn.init.xavier_uniform_(self.bases[b])
        nn.init.xavier_uniform_(self.coef)

    def forward(self, x, edge_index, edge_type, norm):
        W = torch.einsum("rb,bio->rio", self.coef, self.bases)          # [R, in, out]
        out = self.self_loop(x)
        src, dst = edge_index
        for r in range(self.num_rel):
            m = edge_type == r
            if m.any():
                msg = (x[src[m]] @ W[r]) * norm[m][:, None]
                out = out.index_add(0, dst[m], msg)
        return out


class RGCNEncoderLite(nn.Module):
    """node types (8) + learnable embedding (64) -> 2 R-GCN layers -> [N, out_dim]."""

    def __init__(self, num_entities, type_dim=8, emb_dim=64, hidden=128, out_dim=128,
                 num_rel=12, num_bases=4, dropout=0.2):
        super().__init__()
        self.num_rel = num_rel
        self.emb = nn.Embedding(num_entities, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.l1 = RGCNLayer(type_dim + emb_dim, hidden, num_rel, num_bases)
        self.l2 = RGCNLayer(hidden, out_dim, num_rel, num_bases)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_types, edge_index, edge_type):
        N = node_types.size(0)
        key = edge_index[1] * self.num_rel + edge_type
        cnt = torch.bincount(key, minlength=N * self.num_rel).float()
        norm = 1.0 / cnt[key]
        x = torch.cat([node_types, self.emb.weight], dim=1)
        x = self.drop(F.relu(self.l1(x, edge_index, edge_type, norm)))
        return self.l2(x, edge_index, edge_type, norm)


class RGCNDistMultBaseline(nn.Module):
    """R-GCN encoder + DistMult decoder, no variational part. score = sum(z_h * r * z_t)."""

    def __init__(self, data: dict, emb_dim=64, hidden=128, out_dim=128, num_bases=4, dropout=0.2,
                 use_graph=True):
        super().__init__()
        self.N = data["meta"]["num_entities"]
        R = data["meta"]["num_relations"]
        self.register_buffer("node_types", data["node_types"].float())
        self.register_buffer("edge_index", data["edge_index"])
        self.register_buffer("edge_type", data["edge_type"])
        self.encoder = RGCNEncoderLite(self.N, data["node_types"].size(1), emb_dim, hidden,
                                       out_dim, 2 * R, num_bases, dropout)
        self.use_graph = use_graph
        self.rel = nn.Parameter(torch.empty(R, out_dim if use_graph else emb_dim))
        nn.init.xavier_uniform_(self.rel)

    def encode(self, edge_index=None, edge_type=None):
        if not self.use_graph:                     # diagnostic: free embeddings only
            return self.encoder.emb.weight
        ei = self.edge_index if edge_index is None else edge_index
        et = self.edge_type if edge_type is None else edge_type
        return self.encoder(self.node_types, ei, et)

    @staticmethod
    def _score(z, rel, h, r, t):
        return (z[h] * rel[r] * z[t]).sum(-1)

    @torch.no_grad()
    def make_score_all_tails(self):
        self.eval()
        z = self.encode()                                       # full train graph, no masking
        return lambda h, r: (z[h] * self.rel[r]) @ z.T          # [B, N]

    @torch.no_grad()
    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        self.eval()
        z = self.encode()
        return self._score(z, self.rel, triples[:, 0], triples[:, 1], triples[:, 2])

    def fit(self, data, epochs=100, lr=1e-3, num_neg=16, batch_size=2048, adv_temp=1.0,
            eval_every=5, seed=42, weight_decay=1e-5, log=print):
        torch.manual_seed(seed)
        train, val = data["train"], data["val"]
        n = len(train)
        assert self.edge_index.size(1) == 2 * n, "graph must be train edges + inverses"
        tid = torch.arange(n).repeat(2)          # edge k and n+k both come from train triple k
        known = torch.cat([data["train"], data["val"], data["test"]])
        sampler = UniformNegativeSampler(known.numpy(), self.N, data["meta"]["num_relations"], seed=seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_mrr, best_state, best_ep = -1.0, None, 0

        for ep in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(n)
            total = 0.0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                b = train[idx]
                h, r, t = b[:, 0], b[:, 1], b[:, 2]
                B = len(b)
                drop = torch.zeros(n, dtype=torch.bool)
                drop[idx] = True
                keep = ~drop[tid]                        # hide this batch's own edges from the graph
                z = self.encode(self.edge_index[:, keep], self.edge_type[keep])
                pos = self._score(z, self.rel, h, r, t)
                negs = torch.from_numpy(sampler.sample(b.numpy(), k=num_neg)).long()   # shared sampler -> [B, K, 3]
                hn, rn, tn = negs[..., 0], negs[..., 1], negs[..., 2]
                neg = self._score(z, self.rel, hn, rn, tn)
                w = torch.softmax(adv_temp * neg, dim=1).detach()
                loss = -F.logsigmoid(pos).mean() - (w * F.logsigmoid(-neg)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * B
            if ep % eval_every == 0 or ep == epochs:
                res = evaluate_ranking(self.make_score_all_tails(), val, known, self.N)
                log(f"epoch {ep:4d} | loss {total / n:.4f} | val MRR {res['MRR']:.4f} "
                    f"H@10 {res['Hits@10']:.4f}")
                if res["MRR"] > best_mrr:
                    best_mrr, best_ep, best_state = res["MRR"], ep, copy.deepcopy(self.state_dict())
        self.load_state_dict(best_state)
        self.eval()
        log(f"best epoch {best_ep} (val MRR {best_mrr:.4f})")
        return best_ep, best_mrr