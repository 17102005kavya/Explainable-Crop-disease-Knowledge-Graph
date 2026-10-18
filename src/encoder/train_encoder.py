"""Owner: Person 2 (Part B). Standalone training of the encoder with a throwaway DistMult scorer.

Loads data via src.data.loader.load_processed, builds NodeFeatures + RGCNEncoder, trains on train triples with
negatives from UniformNegativeSampler (src/data/prepare_cropdp.py), and reports filtered val MRR & Hits@10 on 500 triples.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

# Ensure repository root is on sys.path when script is executed directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.loader import load_processed
from src.data.prepare_cropdp import UniformNegativeSampler
from src.encoder.node_features import NodeFeatures
from src.encoder.rgcn import RGCNEncoder



class DistMultScorer(nn.Module):
    """Temporary DistMult scorer using base-relation embeddings (ids 0..num_relations-1)."""

    def __init__(self, num_relations: int = 6, dim: int = 128):
        super().__init__()
        self.num_relations = num_relations
        self.dim = dim
        self.rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(self, H: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute DistMult score for triples (h, r, t).

        H: [N, d], h: [B], r: [B], t: [B] -> [B]
        """
        h_emb = H[h]            # [B, d]
        r_emb = self.rel_emb(r) # [B, d]
        t_emb = H[t]            # [B, d]
        return (h_emb * r_emb * t_emb).sum(dim=-1)

    def score_all_tails(self, H: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Compute DistMult score for all candidate tails given (h, r).

        H: [N, d], h: [B], r: [B] -> [B, N]
        """
        q = H[h] * self.rel_emb(r)  # [B, d]
        return q @ H.t()            # [B, N]


def build_known_tails_dict(known_triples: np.ndarray) -> Dict[Tuple[int, int], set]:
    """Map (head, relation) -> set of tails for filtered evaluation."""
    known_tails: Dict[Tuple[int, int], set] = defaultdict(set)
    for h, r, t in known_triples:
        known_tails[(int(h), int(r))].add(int(t))
    return known_tails


@torch.no_grad()
def evaluate_filtered_tail(
    H: torch.Tensor,
    scorer: DistMultScorer,
    val_triples: torch.Tensor,
    known_tails: Dict[Tuple[int, int], set],
    num_eval: int = 500,
    batch_size: int = 100,
) -> Dict[str, float]:
    """Filtered tail-prediction MRR, Hits@1, Hits@3, and Hits@10 on validation triples.

    If num_eval is 0, None, or negative, evaluates the entire validation set of triples.
    """
    scorer.eval()
    if num_eval is not None and num_eval > 0:
        eval_triples = val_triples[:num_eval]
    else:
        eval_triples = val_triples

    ranks = []

    for i in range(0, len(eval_triples), batch_size):
        batch = eval_triples[i:i + batch_size]
        h = batch[:, 0]
        r = batch[:, 1]
        t = batch[:, 2]

        scores = scorer.score_all_tails(H, h, r)  # [B, N]

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

            # Number of entities with strictly higher score + 1
            rank = (row_scores > target_score).sum().item() + 1
            ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks_arr))
    hits1 = float(np.mean(ranks_arr <= 1.0))
    hits3 = float(np.mean(ranks_arr <= 3.0))
    hits10 = float(np.mean(ranks_arr <= 10.0))

    return {
        "mrr": mrr,
        "hits@1": hits1,
        "hits@3": hits3,
        "hits@10": hits10,
        "num_eval": len(ranks),
    }


def train_encoder(
    config_path: str = "configs/default.yaml",
    epochs: Optional[int] = None,
    eval_every: Optional[int] = None,
    num_neg: Optional[int] = None,
    eval_limit: int = 500,
    checkpoint_path: str = "results/encoder_checkpoint.pt",
    log_path: str = "results/encoder_train.log",
    device: str = "cpu",
) -> Tuple[NodeFeatures, RGCNEncoder, DistMultScorer, dict]:

    """Train NodeFeatures + RGCNEncoder with DistMultScorer and log validation metrics."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_dir = cfg.get("data_dir", "data/processed")
    data = load_processed(data_dir, device=device)

    train_triples = data["train"]
    val_triples = data["val"]
    test_triples = data["test"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    node_types = data["node_types"]
    meta = data["meta"]

    num_entities = meta["num_entities"]
    num_base_relations = meta["num_relations"]

    # Filtered negatives dictionary against all known triples across splits
    known_triples = np.concatenate([
        train_triples.cpu().numpy(),
        val_triples.cpu().numpy(),
        test_triples.cpu().numpy(),
    ], axis=0)
    known_tails = build_known_tails_dict(known_triples)
    sampler = UniformNegativeSampler(
        known_triples=known_triples,
        num_entities=num_entities,
        num_relations=num_base_relations,
        seed=seed,
    )

    enc_cfg = cfg["encoder"]
    node_features = NodeFeatures(node_types, emb_dim=enc_cfg.get("emb_dim", 64)).to(device)
    encoder = RGCNEncoder(
        in_dim=node_features.out_dim,
        hidden_dim=enc_cfg.get("hidden_dim", 128),
        out_dim=enc_cfg.get("out_dim", 128),
        num_relations=enc_cfg.get("num_relations", 12),
        num_bases=enc_cfg.get("num_bases", 4),
        dropout=enc_cfg.get("dropout", 0.2),
    ).to(device)

    dec_cfg = cfg.get("decoder", {})
    num_scorer_relations = dec_cfg.get("num_base_relations", num_base_relations)
    scorer = DistMultScorer(num_relations=num_scorer_relations, dim=enc_cfg.get("out_dim", 128)).to(device)

    train_cfg = cfg.get("training", {})
    total_epochs = epochs if epochs is not None else train_cfg.get("epochs", 200)
    evaluation_interval = eval_every if eval_every is not None else train_cfg.get("eval_every", 10)
    k_neg = num_neg if num_neg is not None else train_cfg.get("num_neg", 16)
    lr = float(train_cfg.get("lr", 0.001))

    optimizer = torch.optim.Adam(
        list(node_features.parameters()) + list(encoder.parameters()) + list(scorer.parameters()),
        lr=lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = Path(checkpoint_path).parent
    chk_dir.mkdir(parents=True, exist_ok=True)

    def log_msg(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    log_msg(f"=== Starting Person 2 R-GCN Encoder Training ===")
    log_msg(f"Entities: {num_entities} | Base relations: {num_base_relations} | Train triples: {len(train_triples)}")
    log_msg(f"Epochs: {total_epochs} | Eval every: {evaluation_interval} | Negatives/pos: {k_neg} | LR: {lr}")

    train_triples_np = train_triples.cpu().numpy()
    best_mrr = -1.0
    latest_metrics = {}

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        node_features.train()
        encoder.train()
        scorer.train()
        optimizer.zero_grad()

        # Forward pass on message-passing graph (train-only graph with inverse edges)
        x = node_features()
        H = encoder(x, edge_index, edge_type)

        # Positive triples
        pos_h = train_triples[:, 0]
        pos_r = train_triples[:, 1]
        pos_t = train_triples[:, 2]
        pos_scores = scorer.score(H, pos_h, pos_r, pos_t)

        # Negative triples sampled and filtered against all known triples
        neg_arr = sampler.sample(train_triples_np, k=k_neg)  # [M, k, 3]
        neg_t = torch.from_numpy(neg_arr).to(device).reshape(-1, 3)
        neg_scores = scorer.score(H, neg_t[:, 0], neg_t[:, 1], neg_t[:, 2])

        # Combined BCE loss
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()
        elapsed = time.time() - t0

        if epoch % evaluation_interval == 0 or epoch == total_epochs:
            node_features.eval()
            encoder.eval()
            with torch.no_grad():
                H_eval = encoder(node_features(), edge_index, edge_type)
                latest_metrics = evaluate_filtered_tail(
                    H=H_eval,
                    scorer=scorer,
                    val_triples=val_triples,
                    known_tails=known_tails,
                    num_eval=eval_limit,
                )

            mrr = latest_metrics["mrr"]
            hits10 = latest_metrics["hits@10"]
            log_msg(
                f"Epoch {epoch:03d}/{total_epochs:03d} [{elapsed:.2f}s] - "
                f"Loss: {loss.item():.4f} | Val MRR ({latest_metrics['num_eval']}): {mrr:.4f} | Hits@10: {hits10:.4f}"
            )

            # Checkpoint on best validation MRR
            if mrr > best_mrr:
                best_mrr = mrr
                checkpoint = {
                    "epoch": epoch,
                    "actual_epochs_completed": epoch,
                    "lr": lr,
                    "num_neg": k_neg,
                    "seed": seed,
                    "eval_limit": eval_limit,
                    "node_features_state_dict": node_features.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "scorer_state_dict": scorer.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": latest_metrics,
                    "config": cfg,
                }
                torch.save(checkpoint, checkpoint_path)
                log_msg(f"  -> Saved new best checkpoint (Val MRR: {mrr:.4f}) to {checkpoint_path}")
        else:
            log_msg(f"Epoch {epoch:03d}/{total_epochs:03d} [{elapsed:.2f}s] - Loss: {loss.item():.4f}")

    log_msg(f"Training completed. Checkpoint saved to {checkpoint_path}")
    return node_features, encoder, scorer, latest_metrics


def evaluate_checkpoint(
    checkpoint_path: str = "results/encoder_checkpoint.pt",
    eval_limit: int = 0,
    save_checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate a saved checkpoint on validation triples.

    eval_limit = 0 evaluates all validation triples.
    If save_checkpoint_path is specified, saves updated metadata with the evaluation results
    under a distinct file path without modifying the original checkpoint.
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    cfg = ckpt.get("config", {})
    data_dir = cfg.get("data_dir", "data/processed")
    data = load_processed(data_dir, device=device)

    val_triples = data["val"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    node_types = data["node_types"]

    known = np.concatenate([
        data["train"].cpu().numpy(),
        val_triples.cpu().numpy(),
        data["test"].cpu().numpy(),
    ], axis=0)
    known_tails = build_known_tails_dict(known)

    enc_cfg = cfg.get("encoder", {})
    node_features = NodeFeatures(node_types, emb_dim=enc_cfg.get("emb_dim", 64)).to(device)
    encoder = RGCNEncoder(
        in_dim=node_features.out_dim,
        hidden_dim=enc_cfg.get("hidden_dim", 128),
        out_dim=enc_cfg.get("out_dim", 128),
        num_relations=enc_cfg.get("num_relations", 12),
        num_bases=enc_cfg.get("num_bases", 4),
        dropout=enc_cfg.get("dropout", 0.2),
    ).to(device)
    dec_cfg = cfg.get("decoder", {})
    num_scorer_relations = dec_cfg.get("num_base_relations", data["meta"]["num_relations"])
    scorer = DistMultScorer(num_relations=num_scorer_relations, dim=enc_cfg.get("out_dim", 128)).to(device)

    node_features.load_state_dict(ckpt["node_features_state_dict"])
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    scorer.load_state_dict(ckpt["scorer_state_dict"])

    node_features.eval()
    encoder.eval()
    scorer.eval()

    with torch.no_grad():
        x = node_features()
        H = encoder(x, edge_index, edge_type)
        metrics = evaluate_filtered_tail(
            H=H,
            scorer=scorer,
            val_triples=val_triples,
            known_tails=known_tails,
            num_eval=eval_limit,
        )

    print(f"=== Validation Evaluation (Triples: {metrics['num_eval']}) ===")
    print(f"MRR:     {metrics['mrr']:.4f}")
    print(f"Hits@1:  {metrics['hits@1']:.4f}")
    print(f"Hits@3:  {metrics['hits@3']:.4f}")
    print(f"Hits@10: {metrics['hits@10']:.4f}")

    if save_checkpoint_path is not None:
        new_ckpt = dict(ckpt)
        new_ckpt["actual_epochs_completed"] = ckpt.get("epoch", 50)
        new_ckpt["lr"] = float(cfg.get("training", {}).get("lr", 0.001))
        new_ckpt["num_neg"] = int(cfg.get("training", {}).get("num_neg", 16))
        new_ckpt["seed"] = int(cfg.get("seed", 42))
        new_ckpt["eval_limit"] = eval_limit
        new_ckpt["metrics"] = metrics
        Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_ckpt, save_checkpoint_path)
        print(f"Saved distinct checkpoint with full validation metadata to: {save_checkpoint_path}")

    return metrics


def main(config_path: str = "configs/default.yaml") -> None:
    parser = argparse.ArgumentParser(description="Person 2 R-GCN Encoder Training")
    parser.add_argument("--config", default=config_path, help="Path to YAML configuration")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--eval-every", type=int, default=None, help="Override validation interval")
    parser.add_argument("--num-neg", type=int, default=None, help="Override number of negative samples")
    parser.add_argument("--eval-limit", type=int, default=500, help="Validation triples to evaluate (0 for full set)")
    parser.add_argument("--checkpoint", default="results/encoder_checkpoint.pt", help="Path to save checkpoint")
    parser.add_argument("--log", default="results/encoder_train.log", help="Path to save log file")
    parser.add_argument("--eval-only", action="store_true", help="Only run validation on the specified checkpoint")
    parser.add_argument("--save-eval-checkpoint", default=None, help="Save updated metadata under distinct checkpoint path")
    args, _ = parser.parse_known_args()

    if args.eval_only:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            eval_limit=args.eval_limit,
            save_checkpoint_path=args.save_eval_checkpoint,
        )
    else:
        train_encoder(
            config_path=args.config,
            epochs=args.epochs,
            eval_every=args.eval_every,
            num_neg=args.num_neg,
            eval_limit=args.eval_limit,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
        )


if __name__ == "__main__":
    main()


