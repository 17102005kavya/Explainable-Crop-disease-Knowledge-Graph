"""Phase 2: Systematic Staged Hyperparameter Tuning for R-GCN + DistMult Baseline.

Evaluates on the FULL validation set (1,942 triples) with early stopping based on validation MRR.
Preserves historical results in results/results.csv and saves the tuned checkpoint to results/rgcn_distmult_tuned.pt.
"""
import argparse
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
from src.encoder.train_encoder import DistMultScorer, evaluate_filtered_tail, build_known_tails_dict


def count_parameters(models: List[nn.Module]) -> int:
    return sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in models)


def run_experiment(
    data: Dict[str, Any],
    known_tails: Dict[Tuple[int, int], set],
    lr: float = 0.001,
    emb_dim: int = 128,
    num_neg: int = 16,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    eval_every: int = 5,
    patience: int = 4,  # 4 * 5 = 20 epochs without improvement
    seed: int = 42,
    device: str = "cpu",
    save_checkpoint_path: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train R-GCN + DistMult with early stopping on full validation MRR."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_triples = data["train"]
    val_triples = data["val"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    node_types = data["node_types"]
    meta = data["meta"]
    num_entities = meta["num_entities"]
    num_base_relations = meta["num_relations"]

    known_all = np.concatenate([
        train_triples.cpu().numpy(),
        val_triples.cpu().numpy(),
        data["test"].cpu().numpy(),
    ], axis=0)

    sampler = UniformNegativeSampler(
        known_triples=known_all,
        num_entities=num_entities,
        num_relations=num_base_relations,
        seed=seed,
    )

    # Node features: 8 type features + (emb_dim // 2) learnable embedding
    node_features = NodeFeatures(node_types, emb_dim=emb_dim // 2).to(device)
    encoder = RGCNEncoder(
        in_dim=node_features.out_dim,
        hidden_dim=emb_dim,
        out_dim=emb_dim,
        num_relations=12,
        num_bases=4,
        dropout=dropout,
    ).to(device)
    scorer = DistMultScorer(num_relations=num_base_relations, dim=emb_dim).to(device)

    param_count = count_parameters([node_features, encoder, scorer])

    all_params = list(node_features.parameters()) + list(encoder.parameters()) + list(scorer.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.05)
    criterion = nn.BCEWithLogitsLoss()

    train_triples_np = train_triples.cpu().numpy()
    best_mrr = -1.0
    best_metrics = {}
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    t_start = time.time()

    def log(msg: str):
        print(msg)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    log(f"\n--- Run: lr={lr}, dim={emb_dim}, neg={num_neg}, drop={dropout}, seed={seed} (Params: {param_count:,}) ---")

    for epoch in range(1, max_epochs + 1):
        node_features.train()
        encoder.train()
        scorer.train()
        optimizer.zero_grad()

        x = node_features()
        H = encoder(x, edge_index, edge_type)

        pos_h, pos_r, pos_t = train_triples[:, 0], train_triples[:, 1], train_triples[:, 2]
        pos_scores = scorer.score(H, pos_h, pos_r, pos_t)

        neg_arr = sampler.sample(train_triples_np, k=num_neg)
        neg_t = torch.from_numpy(neg_arr).to(device).reshape(-1, 3)
        neg_scores = scorer.score(H, neg_t[:, 0], neg_t[:, 1], neg_t[:, 2])

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
        loss = criterion(scores, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % eval_every == 0 or epoch == max_epochs:
            node_features.eval()
            encoder.eval()
            scorer.eval()
            with torch.no_grad():
                H_eval = encoder(node_features(), edge_index, edge_type)
                metrics = evaluate_filtered_tail(
                    H=H_eval,
                    scorer=scorer,
                    val_triples=val_triples,
                    known_tails=known_tails,
                    num_eval=0,  # full validation set
                    batch_size=256,
                )

            mrr = metrics["mrr"]
            h10 = metrics["hits@10"]
            current_lr = scheduler.get_last_lr()[0]
            log(f"Epoch {epoch:03d}/{max_epochs:03d} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Val MRR: {mrr:.4f} | Hits@10: {h10:.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                best_metrics = metrics
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {
                    "epoch": epoch,
                    "lr": lr,
                    "emb_dim": emb_dim,
                    "num_neg": num_neg,
                    "dropout": dropout,
                    "seed": seed,
                    "node_features_state_dict": node_features.state_dict(),
                    "encoder_state_dict": encoder.state_dict(),
                    "scorer_state_dict": scorer.state_dict(),
                    "metrics": metrics,
                    "num_params": param_count,
                }
                if save_checkpoint_path:
                    Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state, save_checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log(f"Early stopping triggered at epoch {epoch} (no improvement for {epochs_no_improve * eval_every} epochs).")
                    break

    total_time = time.time() - t_start
    log(f"Finished: Best Epoch {best_epoch} | Best Val MRR: {best_metrics.get('mrr', 0.0):.4f} | Hits@10: {best_metrics.get('hits@10', 0.0):.4f} | Time: {total_time:.1f}s")

    return {
        "best_epoch": best_epoch,
        "mrr": best_metrics.get("mrr", 0.0),
        "hits@1": best_metrics.get("hits@1", 0.0),
        "hits@3": best_metrics.get("hits@3", 0.0),
        "hits@10": best_metrics.get("hits@10", 0.0),
        "total_time": total_time,
        "param_count": param_count,
        "best_state": best_state,
    }


def append_to_results(
    csv_path: str,
    run_id: str,
    model: str,
    config_desc: str,
    seed: int,
    metrics: Dict[str, float],
    notes: str,
):
    df_row = pd.DataFrame([{
        "run_id": run_id,
        "date": date.today().isoformat(),
        "owner": "Person 2",
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


def run_staged_tuning():
    log_file = Path("results/rgcn_distmult_tuning.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.unlink()

    data = load_processed("data/processed")
    known_all = np.concatenate([
        data["train"].cpu().numpy(),
        data["val"].cpu().numpy(),
        data["test"].cpu().numpy(),
    ], axis=0)
    known_tails = build_known_tails_dict(known_all)

    results_csv = "results/results.csv"

    print("=================================================================")
    print("PHASE 2: STAGED HYPERPARAMETER TUNING FOR R-GCN + DISTMULT")
    print("=================================================================")

    # -------------------------------------------------------------
    # STAGE 2.1: Learning Rate Search (0.0001, 0.0005, 0.001, 0.003)
    # Default: dim=128, neg=16, drop=0.2
    # -------------------------------------------------------------
    print("\n--- STAGE 2.1: Learning Rate Search ---")
    lr_candidates = [0.0001, 0.0005, 0.001, 0.003]
    stage1_results = {}
    for lr in lr_candidates:
        res = run_experiment(
            data=data,
            known_tails=known_tails,
            lr=lr,
            emb_dim=128,
            num_neg=16,
            dropout=0.2,
            max_epochs=80,
            eval_every=5,
            patience=4,
            seed=42,
            log_file=log_file,
        )
        stage1_results[lr] = res
        append_to_results(
            results_csv,
            run_id=f"phase2_tune_lr_{lr}",
            model="RGCN+DistMult",
            config_desc=f"lr={lr},dim=128,neg=16,drop=0.2",
            seed=42,
            metrics=res,
            notes=f"Stage 2.1 LR tuning (best epoch {res['best_epoch']})",
        )

    best_lr = max(stage1_results.keys(), key=lambda k: stage1_results[k]["mrr"])
    print(f"\n>>> Best LR from Stage 2.1: {best_lr} (Val MRR: {stage1_results[best_lr]['mrr']:.4f})")

    # -------------------------------------------------------------
    # STAGE 2.2: Negative Samples Search (16, 32, 64)
    # -------------------------------------------------------------
    print("\n--- STAGE 2.2: Negative Ratio Search ---")
    neg_candidates = [16, 32, 64]
    stage2_results = {}
    for neg in neg_candidates:
        if neg == 16:
            stage2_results[neg] = stage1_results[best_lr]
            continue
        res = run_experiment(
            data=data,
            known_tails=known_tails,
            lr=best_lr,
            emb_dim=128,
            num_neg=neg,
            dropout=0.2,
            max_epochs=80,
            eval_every=5,
            patience=4,
            seed=42,
            log_file=log_file,
        )
        stage2_results[neg] = res
        append_to_results(
            results_csv,
            run_id=f"phase2_tune_neg_{neg}",
            model="RGCN+DistMult",
            config_desc=f"lr={best_lr},dim=128,neg={neg},drop=0.2",
            seed=42,
            metrics=res,
            notes=f"Stage 2.2 Negatives tuning (best epoch {res['best_epoch']})",
        )

    best_neg = max(stage2_results.keys(), key=lambda k: stage2_results[k]["mrr"])
    print(f"\n>>> Best Negative Ratio from Stage 2.2: {best_neg} (Val MRR: {stage2_results[best_neg]['mrr']:.4f})")

    # -------------------------------------------------------------
    # STAGE 2.3: Dropout (0.1, 0.2, 0.3) & Dimension (128, 256)
    # -------------------------------------------------------------
    print("\n--- STAGE 2.3: Dropout & Dimension Search ---")
    reg_configs = [
        {"drop": 0.1, "dim": 128},
        {"drop": 0.2, "dim": 128},
        {"drop": 0.3, "dim": 128},
        {"drop": 0.2, "dim": 256},
    ]
    stage3_results = {}
    for cfg in reg_configs:
        drop, dim = cfg["drop"], cfg["dim"]
        if drop == 0.2 and dim == 128:
            stage3_results[(drop, dim)] = stage2_results[best_neg]
            continue
        res = run_experiment(
            data=data,
            known_tails=known_tails,
            lr=best_lr,
            emb_dim=dim,
            num_neg=best_neg,
            dropout=drop,
            max_epochs=80,
            eval_every=5,
            patience=4,
            seed=42,
            log_file=log_file,
        )
        stage3_results[(drop, dim)] = res
        append_to_results(
            results_csv,
            run_id=f"phase2_tune_drop{drop}_dim{dim}",
            model="RGCN+DistMult",
            config_desc=f"lr={best_lr},dim={dim},neg={best_neg},drop={drop}",
            seed=42,
            metrics=res,
            notes=f"Stage 2.3 Dropout/Dim tuning (best epoch {res['best_epoch']})",
        )

    best_reg = max(stage3_results.keys(), key=lambda k: stage3_results[k]["mrr"])
    best_drop, best_dim = best_reg
    print(f"\n>>> Best Reg/Dim from Stage 2.3: Drop={best_drop}, Dim={best_dim} (Val MRR: {stage3_results[best_reg]['mrr']:.4f})")

    # -------------------------------------------------------------
    # STAGE 2.4: Multi-Seed Evaluation on Best Configuration
    # Seeds: 42, 123, 999
    # -------------------------------------------------------------
    print("\n--- STAGE 2.4: Multi-Seed Evaluation on Best Config ---")
    seeds = [42, 123, 999]
    seed_results = []
    best_overall_mrr = -1.0
    best_overall_state = None

    for seed in seeds:
        save_path = f"results/rgcn_distmult_seed_{seed}.pt"
        if seed == 42 and best_reg == (0.2, 128) and best_neg == 16 and best_lr == 0.001:
            res = stage1_results[best_lr]
        else:
            res = run_experiment(
                data=data,
                known_tails=known_tails,
                lr=best_lr,
                emb_dim=best_dim,
                num_neg=best_neg,
                dropout=best_drop,
                max_epochs=100,
                eval_every=5,
                patience=5,
                seed=seed,
                save_checkpoint_path=save_path,
                log_file=log_file,
            )
        seed_results.append(res)
        append_to_results(
            results_csv,
            run_id=f"phase2_best_seed_{seed}",
            model="RGCN+DistMult",
            config_desc=f"lr={best_lr},dim={best_dim},neg={best_neg},drop={best_drop}",
            seed=seed,
            metrics=res,
            notes=f"Final selected R-GCN+DistMult seed {seed}",
        )
        if res["mrr"] > best_overall_mrr and res.get("best_state"):
            best_overall_mrr = res["mrr"]
            best_overall_state = res["best_state"]

    if best_overall_state:
        torch.save(best_overall_state, "results/rgcn_distmult_tuned.pt")
        print(f"\n>>> Saved best tuned checkpoint across seeds to results/rgcn_distmult_tuned.pt (Val MRR: {best_overall_mrr:.4f})")

    mrrs = [r["mrr"] for r in seed_results]
    h10s = [r["hits@10"] for r in seed_results]
    print(f"\nMulti-Seed Summary ({len(seeds)} seeds):")
    print(f"Val MRR:     {np.mean(mrrs):.4f} +/- {np.std(mrrs):.4f}")
    print(f"Val Hits@10: {np.mean(h10s):.4f} +/- {np.std(h10s):.4f}")


if __name__ == "__main__":
    run_staged_tuning()
