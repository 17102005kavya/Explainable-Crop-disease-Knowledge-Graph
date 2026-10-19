"""Owner: Person 4 (Part D). Plug the team's VRGAE into the shared runner and evaluator.

source: checkpoint -> load a checkpoint trained by Person 3 and evaluate it as it is
source: train      -> train the VRGAE with the SAME recipe as the baselines
                      (mini-batches, the batch's own edges hidden from message passing)
The VRGAE import is lazy, so the baselines keep working while main's latent.py / loss.py are stubs.
"""
import copy
import inspect

import torch
import torch.nn.functional as F

from src.eval.evaluator import evaluate_ranking


class VRGAEScorer:
    """Wraps a VRGAE and exposes what the evaluator needs. Uses z = mu and the full TRAIN graph."""

    def __init__(self, model, edge_index, edge_type, chunk_size: int = 32):
        self.model, self.edge_index, self.edge_type, self.chunk = model, edge_index, edge_type, chunk_size

    @torch.no_grad()
    def _z(self):
        self.model.eval()
        _, mu, _ = self.model.encode(self.edge_index, self.edge_type)
        return mu

    @torch.no_grad()
    def make_score_all_tails(self):
        z = self._z()
        fn = self.model.decoder.score_all_tails
        kw = {"chunk_size": self.chunk} if "chunk_size" in inspect.signature(fn).parameters else {}
        return lambda h, r: fn(z, h, r, **kw)

    @torch.no_grad()
    def score_triples(self, triples: torch.Tensor) -> torch.Tensor:
        z = self._z()
        return self.model.score(z, triples[:, 0], triples[:, 1], triples[:, 2])


def fit_shared_recipe(model, data: dict, epochs=100, lr=0.001, weight_decay=0.0001, num_neg=16,
                      batch_size=2048, adv_temp=1.0, kl_weight=0.001, kl_warmup=20,
                      eval_every=5, seed=42, chunk_size=32, log=print):
    """Mini-batch training; best epoch picked on the FULL validation split with evaluate_ranking."""
    torch.manual_seed(seed)
    train, val = data["train"], data["val"]
    ei, et = data["edge_index"], data["edge_type"]
    n, N = len(train), data["meta"]["num_entities"]
    assert ei.size(1) == 2 * n, "graph must be train edges + inverses"
    tid = torch.arange(n).repeat(2)                 # edge k and n+k both come from train triple k
    known = torch.cat([data["train"], data["val"], data["test"]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scorer = VRGAEScorer(model, ei, et, chunk_size)
    best_mrr, best_state, best_ep = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        beta = kl_weight * min(1.0, ep / kl_warmup) if kl_warmup > 0 else kl_weight
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            b = train[idx]
            h, r, t = b[:, 0], b[:, 1], b[:, 2]
            B = len(b)
            drop = torch.zeros(n, dtype=torch.bool)
            drop[idx] = True
            keep = ~drop[tid]                        # hide this batch's own edges from the graph
            z, mu, logvar = model.encode(ei[:, keep], et[keep])
            pos = model.score(z, h, r, t)
            rand = torch.randint(0, N, (B, num_neg))
            head_side = torch.rand(B, 1) < 0.5
            hn = torch.where(head_side, rand, h[:, None].expand(B, num_neg))
            tn = torch.where(head_side, t[:, None].expand(B, num_neg), rand)
            rn = r[:, None].expand(B, num_neg)
            neg = model.score(z, hn.reshape(-1), rn.reshape(-1), tn.reshape(-1)).view(B, num_neg)
            w = torch.softmax(adv_temp * neg, dim=1).detach()
            link = -F.logsigmoid(pos).mean() - (w * F.logsigmoid(-neg)).sum(1).mean()
            loss = link + beta * model.latent.kl_divergence(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * B
        if ep % eval_every == 0 or ep == epochs:
            res = evaluate_ranking(scorer.make_score_all_tails(), val, known, N)
            log(f"epoch {ep:4d} | loss {total / n:.4f} | val MRR {res['MRR']:.4f} H@10 {res['Hits@10']:.4f}")
            if res["MRR"] > best_mrr:
                best_mrr, best_ep, best_state = res["MRR"], ep, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    log(f"best epoch {best_ep} (val MRR {best_mrr:.4f})")
    return best_ep, best_mrr


def build_vrgae(cfg: dict, d: dict):
    """Called by run_experiment for model: vrgae. Returns (scorer, extra notes)."""
    from src.model.vrgae import VRGAE               # lazy: needs the real latent.py / loss.py on main

    seed = cfg["seed"]
    torch.manual_seed(seed)
    model_cfg = {k: cfg[k] for k in ("encoder", "latent", "decoder") if k in cfg}
    model_cfg["model"] = cfg.get("architecture", {"decoder": "rotate"})   # VRGAE expects cfg["model"]["decoder"]
    model = VRGAE(d["node_types"], model_cfg)
    chunk = cfg.get("eval_chunk", 32)

    if cfg.get("source", "train") == "checkpoint":
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state") or ckpt["model_state_dict"])
        extra = {"checkpoint": cfg["checkpoint"], "ckpt_epoch": ckpt.get("epoch")}
    else:
        ep, mrr = fit_shared_recipe(model, d, seed=seed, chunk_size=chunk, **cfg.get("training", {}))
        extra = {"best_epoch": ep, "val_MRR": round(mrr, 4)}
    model.eval()
    return VRGAEScorer(model, d["edge_index"], d["edge_type"], chunk), extra