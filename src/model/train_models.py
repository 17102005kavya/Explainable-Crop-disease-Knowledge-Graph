"""Phase 3 & 4: Training and Evaluation Pipeline for ComplEx, RotatE, and Full VRGAE.

Evaluates all models using identical splits, identical candidate pool (N=11,417),
and identical filtered tail-prediction protocol.
Reports overall and per-relation metrics (especially ManifestAs).
Saves distinct checkpoints and appends measured results to results/results.csv.
"""
from datetime import date
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.loader import load_processed
from src.data.prepare_cropdp import UniformNegativeSampler
from src.encoder.node_features import NodeFeatures
from src.encoder.rgcn import RGCNEncoder
from src.model.decoder import ComplExDecoder, RotatEDecoder
from src.model.latent import VariationalLatent
from src.model.loss import link_loss, total_loss
from src.model.vrgae import VRGAE
from src.eval.evaluator import evaluate_ranking, build_known_tails_dict


def count_params(model_or_list) -> int:
    if isinstance(model_or_list, list):
        return sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in model_or_list)
    return sum(p.numel() for p in model_or_list.parameters() if p.requires_grad)


def append_result(
    csv_path: str,
    run_id: str,
    model: str,
    config_desc: str,
    seed: int,
    metrics: Dict[str, Any],
    notes: str,
):
    df_row = pd.DataFrame([{
        "run_id": run_id,
        "date": date.today().isoformat(),
        "owner": "Person 3",
        "model": model,
        "config": config_desc,
        "seed": seed,
        "relation": "all",
        "MRR": round(metrics["mrr"], 4),
        "Hits@1": round(metrics["hits@1"], 4),
        "Hits@3": round(metrics["hits@3"], 4),
        "Hits@10": round(metrics["hits@10"], 4),
        "AUC": "",
        "notes": notes,
    }])
    if Path(csv_path).exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def train_model(
    model_type: str = "complex",  # "complex", "rotate", "vrgae"
    lr: float = 0.003,
    emb_dim: int = 256,
    num_neg: int = 32,
    dropout: float = 0.2,
    gamma: float = 6.0,
    max_epochs: int = 80,
    eval_every: int = 5,
    patience: int = 5,
    seed: int = 42,
    device: str = "cpu",
    save_checkpoint_path: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train ComplEx, RotatE, or VRGAE with early stopping on full validation MRR."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_processed("data/processed", device=device)
    train_triples = data["train"]
    val_triples = data["val"]
    test_triples = data["test"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    node_types = data["node_types"]
    meta = data["meta"]
    num_entities = meta["num_entities"]
    num_base_relations = meta["num_relations"]
    rel_names = meta["relations"]

    known_all = torch.cat([train_triples, val_triples, test_triples], dim=0)

    sampler = UniformNegativeSampler(
        known_triples=known_all.cpu().numpy(),
        num_entities=num_entities,
        num_relations=num_base_relations,
        seed=seed,
    )

    def log(msg: str):
        print(msg)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    log(f"\n=======================================================")
    log(f"Training Model: {model_type.upper()} (Seed: {seed}, Dim: {emb_dim}, LR: {lr}, Neg: {num_neg})")
    log(f"=======================================================")

    if model_type == "complex":
        node_features = NodeFeatures(node_types, emb_dim=emb_dim // 2).to(device)
        encoder = RGCNEncoder(
            in_dim=node_features.out_dim,
            hidden_dim=emb_dim,
            out_dim=emb_dim,
            num_relations=12,
            num_bases=4,
            dropout=dropout,
        ).to(device)
        decoder = ComplExDecoder(num_base_relations=num_base_relations, dim=emb_dim).to(device)
        model_parts = [node_features, encoder, decoder]
        param_count = count_params(model_parts)
        optimizer = torch.optim.AdamW(
            list(node_features.parameters()) + list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

        def forward_and_score(pos_batch, neg_batch):
            x = node_features()
            H = encoder(x, edge_index, edge_type)
            pos_s = decoder.score(H, pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2])
            neg_s = decoder.score(H, neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            return link_loss(pos_s, neg_s)

        def make_eval_scorer():
            node_features.eval()
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                H_eval = encoder(node_features(), edge_index, edge_type)
            return lambda h, r: decoder.score_all_tails(H_eval, h, r)

    elif model_type == "rotate":
        node_features = NodeFeatures(node_types, emb_dim=emb_dim // 2).to(device)
        encoder = RGCNEncoder(
            in_dim=node_features.out_dim,
            hidden_dim=emb_dim,
            out_dim=emb_dim,
            num_relations=12,
            num_bases=4,
            dropout=dropout,
        ).to(device)
        decoder = RotatEDecoder(num_base_relations=num_base_relations, dim=emb_dim, gamma=gamma).to(device)
        model_parts = [node_features, encoder, decoder]
        param_count = count_params(model_parts)
        optimizer = torch.optim.AdamW(
            list(node_features.parameters()) + list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )

        def forward_and_score(pos_batch, neg_batch):
            x = node_features()
            H = encoder(x, edge_index, edge_type)
            pos_s = decoder.score(H, pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2])
            neg_s = decoder.score(H, neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            return link_loss(pos_s, neg_s)

        def make_eval_scorer():
            node_features.eval()
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                H_eval = encoder(node_features(), edge_index, edge_type)
            return lambda h, r: decoder.score_all_tails(H_eval, h, r, chunk_size=100)

    elif model_type == "vrgae":
        cfg = {
            "encoder": {
                "emb_dim": emb_dim // 2,
                "hidden_dim": emb_dim,
                "out_dim": emb_dim,
                "num_relations": 12,
                "num_bases": 4,
                "dropout": dropout,
            },
            "decoder": {
                "num_base_relations": num_base_relations,
                "gamma": gamma,
            },
            "model": {
                "decoder": "rotate",
            },
        }
        vrgae = VRGAE(node_types, cfg).to(device)
        model_parts = [vrgae]
        param_count = count_params(vrgae)
        optimizer = torch.optim.AdamW(vrgae.parameters(), lr=lr, weight_decay=1e-4)

        def forward_and_score(pos_batch, neg_batch, epoch_idx):
            z, mu, logvar = vrgae(edge_index, edge_type)
            pos_s = vrgae.score(z, pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2])
            neg_s = vrgae.score(z, neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
            return total_loss(pos_s, neg_s, mu, logvar, epoch=epoch_idx, warmup_epochs=20, max_kl_weight=0.001)

        def make_eval_scorer():
            return vrgae.make_score_all_tails(edge_index, edge_type)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    train_triples_np = train_triples.cpu().numpy()

    best_mrr = -1.0
    best_metrics = {}
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        for p in model_parts:
            p.train()
        optimizer.zero_grad()

        neg_arr = sampler.sample(train_triples_np, k=num_neg)
        neg_t = torch.from_numpy(neg_arr).to(device).reshape(-1, 3)

        if model_type == "vrgae":
            loss = forward_and_score(train_triples, neg_t, epoch)
        else:
            loss = forward_and_score(train_triples, neg_t)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % eval_every == 0 or epoch == max_epochs:
            score_fn = make_eval_scorer()
            metrics = evaluate_ranking(
                score_all_tails=score_fn,
                triples=val_triples,
                known_triples=known_all,
                num_entities=num_entities,
                batch_size=256,
                relation_names=rel_names,
            )

            mrr = metrics["mrr"]
            h10 = metrics["hits@10"]
            manifest_mrr = metrics["per_relation"].get(4, {}).get("mrr", 0.0)
            manifest_h10 = metrics["per_relation"].get(4, {}).get("hits@10", 0.0)

            log(
                f"Epoch {epoch:03d}/{max_epochs:03d} | Loss: {loss.item():.4f} | "
                f"Val MRR: {mrr:.4f} | Hits@10: {h10:.4f} | "
                f"ManifestAs MRR: {manifest_mrr:.4f} (H@10: {manifest_h10:.4f})"
            )

            if mrr > best_mrr:
                best_mrr = mrr
                best_metrics = metrics
                best_epoch = epoch
                epochs_no_improve = 0
                if save_checkpoint_path:
                    Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    if model_type == "vrgae":
                        state = {"model_state": vrgae.state_dict(), "metrics": metrics, "epoch": epoch}
                    elif model_type in ("complex", "rotate"):
                        state = {
                            "node_features": node_features.state_dict(),
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "metrics": metrics,
                            "epoch": epoch,
                        }
                    torch.save(state, save_checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log(f"Early stopping at epoch {epoch} (no improvement for {epochs_no_improve * eval_every} epochs).")
                    break

    total_time = time.time() - t_start
    log(f"Finished {model_type}: Best Epoch {best_epoch} | Best Val MRR: {best_metrics.get('mrr', 0.0):.4f} | Hits@10: {best_metrics.get('hits@10', 0.0):.4f} | Time: {total_time:.1f}s")

    return {
        "model": model_type,
        "best_epoch": best_epoch,
        "mrr": best_metrics.get("mrr", 0.0),
        "hits@1": best_metrics.get("hits@1", 0.0),
        "hits@3": best_metrics.get("hits@3", 0.0),
        "hits@10": best_metrics.get("hits@10", 0.0),
        "total_time": total_time,
        "param_count": param_count,
        "per_relation": best_metrics.get("per_relation", {}),
    }


def run_all_models():
    log_file = Path("results/models_comparison.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    results_csv = "results/results.csv"

    # 1. ComplEx Model
    res_complex = train_model(
        model_type="complex",
        lr=0.003,
        emb_dim=256,
        num_neg=32,
        dropout=0.2,
        max_epochs=80,
        eval_every=5,
        patience=5,
        seed=42,
        save_checkpoint_path="results/rgcn_complex.pt",
        log_file=log_file,
    )
    append_result(
        results_csv,
        run_id="phase3_rgcn_complex",
        model="RGCN+ComplEx",
        config_desc="lr=0.003,dim=256,neg=32,drop=0.2",
        seed=42,
        metrics=res_complex,
        notes=f"ComplEx decoder (best epoch {res_complex['best_epoch']})",
    )

    # 2. RotatE Model
    res_rotate = train_model(
        model_type="rotate",
        lr=0.003,
        emb_dim=256,
        num_neg=32,
        dropout=0.2,
        gamma=6.0,
        max_epochs=80,
        eval_every=5,
        patience=5,
        seed=42,
        save_checkpoint_path="results/rgcn_rotate.pt",
        log_file=log_file,
    )
    append_result(
        results_csv,
        run_id="phase3_rgcn_rotate",
        model="RGCN+RotatE",
        config_desc="lr=0.003,dim=256,neg=32,drop=0.2,gamma=6.0",
        seed=42,
        metrics=res_rotate,
        notes=f"RotatE decoder (best epoch {res_rotate['best_epoch']})",
    )

    # 3. Full VRGAE Model
    res_vrgae = train_model(
        model_type="vrgae",
        lr=0.003,
        emb_dim=256,
        num_neg=32,
        dropout=0.2,
        gamma=6.0,
        max_epochs=80,
        eval_every=5,
        patience=5,
        seed=42,
        save_checkpoint_path="results/vrgae_best.pt",
        log_file=log_file,
    )
    append_result(
        results_csv,
        run_id="phase3_vrgae_full",
        model="VRGAE (Full)",
        config_desc="lr=0.003,dim=256,neg=32,drop=0.2,gamma=6.0,beta=0.001",
        seed=42,
        metrics=res_vrgae,
        notes=f"Complete VRGAE (best epoch {res_vrgae['best_epoch']})",
    )


if __name__ == "__main__":
    run_all_models()
