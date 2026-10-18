"""Owner: Person 3 (Part C). Training loop for the full VRGAE model.

Loads processed data via src.data.loader.load_processed, constructs the integrated VRGAE model,
trains with UniformNegativeSampler and composite loss (link prediction + KL warmup),
evaluates filtered MRR & Hits@10 on validation triples, and saves best checkpoints.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

from src.data.loader import load_processed
from src.data.prepare_cropdp import UniformNegativeSampler
from src.model.vrgae import VRGAE


def build_known_tails_dict(known_triples: np.ndarray) -> Dict[Tuple[int, int], set]:
    """Map (head, relation) -> set of tails for filtered evaluation."""
    known_tails: Dict[Tuple[int, int], set] = defaultdict(set)
    for h, r, t in known_triples:
        known_tails[(int(h), int(r))].add(int(t))
    return known_tails


@torch.no_grad()
def evaluate_vrgae_filtered_tail(
    model: VRGAE,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    val_triples: torch.Tensor,
    known_tails: Dict[Tuple[int, int], set],
    num_eval: int = 500,
    batch_size: int = 100,
) -> Dict[str, float]:
    """Filtered tail-prediction MRR, Hits@1, Hits@3, and Hits@10 on validation triples for VRGAE."""
    model.eval()
    score_fn = model.make_score_all_tails(edge_index, edge_type)

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

        scores = score_fn(h, r)  # [B, N]

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


def train_vrgae(
    config_path: str = "configs/default.yaml",
    epochs: Optional[int] = None,
    eval_every: Optional[int] = None,
    num_neg: Optional[int] = None,
    eval_limit: int = 500,
    checkpoint_path: str = "results/vrgae_checkpoint.pt",
    log_path: str = "results/vrgae_train.log",
    device: str = "cpu",
) -> Tuple[VRGAE, dict]:
    """Train the full VRGAE model on CropDP-KG and log validation metrics."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
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

    model = VRGAE(node_types=node_types, cfg=cfg).to(device)

    train_cfg = cfg.get("training", {})
    total_epochs = epochs if epochs is not None else train_cfg.get("epochs", 200)
    evaluation_interval = eval_every if eval_every is not None else train_cfg.get("eval_every", 10)
    k_neg = num_neg if num_neg is not None else train_cfg.get("num_neg", 16)
    lr = float(train_cfg.get("lr", 0.001))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = Path(checkpoint_path).parent
    chk_dir.mkdir(parents=True, exist_ok=True)

    def log_msg(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    log_msg("=== Starting Person 3 VRGAE Training ===")
    log_msg(f"Entities: {num_entities} | Base relations: {num_base_relations} | Train triples: {len(train_triples)}")
    log_msg(f"Epochs: {total_epochs} | Eval every: {evaluation_interval} | Negatives/pos: {k_neg} | LR: {lr}")

    train_triples_np = train_triples.cpu().numpy()
    best_mrr = -1.0
    latest_metrics = {}

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        # Forward pass through encoder and latent layer
        z, mu, logvar = model(edge_index, edge_type)

        # Sample filtered negative triples
        neg_arr = sampler.sample(train_triples_np, k=k_neg)
        neg_triples = torch.from_numpy(neg_arr).to(device).reshape(-1, 3)

        # Compute composite loss (link prediction + KL warm-up)
        loss, loss_dict = model.compute_loss(
            z=z,
            mu=mu,
            logvar=logvar,
            pos_triples=train_triples,
            neg_triples=neg_triples,
            epoch=epoch,
        )

        loss.backward()
        optimizer.step()
        elapsed = time.time() - t0

        if epoch % evaluation_interval == 0 or epoch == total_epochs:
            latest_metrics = evaluate_vrgae_filtered_tail(
                model=model,
                edge_index=edge_index,
                edge_type=edge_type,
                val_triples=val_triples,
                known_tails=known_tails,
                num_eval=eval_limit,
            )

            mrr = latest_metrics["mrr"]
            hits10 = latest_metrics["hits@10"]
            log_msg(
                f"Epoch {epoch:03d}/{total_epochs:03d} [{elapsed:.2f}s] - "
                f"Loss: {loss_dict['loss']:.4f} (Link: {loss_dict['link_loss']:.4f}, "
                f"KL: {loss_dict['kl_loss']:.4f}, Beta: {loss_dict['kl_weight']:.3f}) | "
                f"Val MRR ({latest_metrics['num_eval']}): {mrr:.4f} | Hits@10: {hits10:.4f}"
            )

            if mrr > best_mrr:
                best_mrr = mrr
                checkpoint = {
                    "epoch": epoch,
                    "actual_epochs_completed": epoch,
                    "lr": lr,
                    "num_neg": k_neg,
                    "seed": seed,
                    "eval_limit": eval_limit,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": latest_metrics,
                    "config": cfg,
                }
                torch.save(checkpoint, checkpoint_path)
                log_msg(f"  -> Saved new best checkpoint (Val MRR: {mrr:.4f}) to {checkpoint_path}")
        else:
            log_msg(
                f"Epoch {epoch:03d}/{total_epochs:03d} [{elapsed:.2f}s] - "
                f"Loss: {loss_dict['loss']:.4f} (Link: {loss_dict['link_loss']:.4f}, KL: {loss_dict['kl_loss']:.4f})"
            )

    log_msg(f"VRGAE training completed. Checkpoint saved to {checkpoint_path}")
    return model, latest_metrics


def main(config_path: str = "configs/default.yaml") -> None:
    parser = argparse.ArgumentParser(description="Person 3 VRGAE Training")
    parser.add_argument("--config", default=config_path, help="Path to YAML configuration")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--eval-every", type=int, default=None, help="Override validation interval")
    parser.add_argument("--num-neg", type=int, default=None, help="Override number of negative samples")
    parser.add_argument("--eval-limit", type=int, default=500, help="Validation triples to evaluate (0 for full set)")
    parser.add_argument("--checkpoint", default="results/vrgae_checkpoint.pt", help="Path to save checkpoint")
    parser.add_argument("--log", default="results/vrgae_train.log", help="Path to save log file")
    args, _ = parser.parse_known_args()

    train_vrgae(
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
