"""Phase 4 & 5: Detailed per-relation, type-constrained ablation, and single-pass test evaluation.

Evaluates saved checkpoints:
1. Historical Baseline (results/encoder_checkpoint_full_val.pt)
2. Tuned R-GCN + DistMult (results/rgcn_distmult_tuned.pt)
3. R-GCN + ComplEx (results/rgcn_complex.pt)
4. R-GCN + RotatE (results/rgcn_rotate.pt)
5. Full VRGAE (results/vrgae_best.pt)
"""
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from src.data.loader import load_processed
from src.encoder.node_features import NodeFeatures
from src.encoder.rgcn import RGCNEncoder
from src.encoder.train_encoder import DistMultScorer
from src.model.decoder import ComplExDecoder, RotatEDecoder
from src.model.vrgae import VRGAE
from src.eval.evaluator import evaluate_ranking, build_known_tails_dict


@torch.no_grad()
def evaluate_type_constrained(
    score_all_tails,
    triples: torch.Tensor,
    known_triples: torch.Tensor,
    node_types: torch.Tensor,
    type_names: List[str],
    rel_names: List[str],
    rel_to_tail_type: Dict[str, str],
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Auxiliary ablation: Evaluate ranking constrained only to candidate entities of the expected tail type."""
    known_tails = build_known_tails_dict(known_triples)
    primary_types = node_types.argmax(dim=-1).cpu().numpy()
    type_to_idx = {name: i for i, name in enumerate(type_names)}

    ranks = []
    ranks_by_rel = defaultdict(list)

    num_eval = len(triples)
    for i in range(0, num_eval, batch_size):
        batch = triples[i:i + batch_size]
        h = batch[:, 0]
        r = batch[:, 1]
        t = batch[:, 2]

        scores = score_all_tails(h, r)

        for j in range(len(batch)):
            hj = int(h[j])
            rj = int(r[j])
            tj = int(t[j])
            rel_name = rel_names[rj]
            expected_type_name = rel_to_tail_type.get(rel_name, "Symptom")
            expected_type_id = type_to_idx.get(expected_type_name, 2)

            target_score = scores[j, tj].item()
            row_scores = scores[j].clone()

            # Mask out any candidate not belonging to the expected tail type
            non_type_mask = primary_types != expected_type_id
            row_scores[non_type_mask] = -1e9

            # Mask out other known true tails
            for ft in known_tails.get((hj, rj), set()):
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
    }

    per_relation = {}
    for rj, r_ranks in ranks_by_rel.items():
        arr = np.array(r_ranks, dtype=np.float64)
        name = rel_names[rj] if rj < len(rel_names) else str(rj)
        per_relation[rj] = {
            "name": name,
            "count": len(arr),
            "mrr": float(np.mean(1.0 / arr)),
            "hits@1": float(np.mean(arr <= 1.0)),
            "hits@3": float(np.mean(arr <= 3.0)),
            "hits@10": float(np.mean(arr <= 10.0)),
        }
    overall["per_relation"] = per_relation
    return overall


def main():
    data = load_processed("data/processed")
    val_triples = data["val"]
    test_triples = data["test"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    node_types = data["node_types"]
    meta = data["meta"]
    num_entities = meta["num_entities"]
    rel_names = meta["relations"]
    type_names = meta["types"]

    known_all = torch.cat([data["train"], val_triples, test_triples], dim=0)

    rel_to_tail_type = {
        "RegionIs": "Area",
        "ConditionIs": "Condition",
        "Damage": "Crop",
        "InflictHarmPart": "Part",
        "ManifestAs": "Symptom",
        "TemperIs": "Temperature",
    }

    models_info = {}

    # 1. Historical Baseline
    if Path("results/encoder_checkpoint_full_val.pt").exists():
        ckpt_base = torch.load("results/encoder_checkpoint_full_val.pt", weights_only=False)
        nf_base = NodeFeatures(node_types, emb_dim=64)
        enc_base = RGCNEncoder(nf_base.out_dim, 128, 128, 12, 4, 0.2)
        sc_base = DistMultScorer(6, 128)
        nf_base.load_state_dict(ckpt_base["node_features_state_dict"])
        enc_base.load_state_dict(ckpt_base["encoder_state_dict"])
        sc_base.load_state_dict(ckpt_base["scorer_state_dict"])
        nf_base.eval(); enc_base.eval(); sc_base.eval()
        with torch.no_grad():
            H_base = enc_base(nf_base(), edge_index, edge_type)
        models_info["Baseline (R-GCN+DistMult, 50ep)"] = lambda h, r: sc_base.score_all_tails(H_base, h, r)

    # 2. Tuned R-GCN + DistMult
    if Path("results/rgcn_distmult_tuned.pt").exists():
        ckpt_tuned = torch.load("results/rgcn_distmult_tuned.pt", weights_only=False)
        emb_dim = ckpt_tuned.get("emb_dim", 256)
        nf_tuned = NodeFeatures(node_types, emb_dim=emb_dim // 2)
        enc_tuned = RGCNEncoder(nf_tuned.out_dim, emb_dim, emb_dim, 12, 4, ckpt_tuned.get("dropout", 0.2))
        sc_tuned = DistMultScorer(6, emb_dim)
        nf_tuned.load_state_dict(ckpt_tuned["node_features_state_dict"])
        enc_tuned.load_state_dict(ckpt_tuned["encoder_state_dict"])
        sc_tuned.load_state_dict(ckpt_tuned["scorer_state_dict"])
        nf_tuned.eval(); enc_tuned.eval(); sc_tuned.eval()
        with torch.no_grad():
            H_tuned = enc_tuned(nf_tuned(), edge_index, edge_type)
        models_info["Tuned R-GCN+DistMult"] = lambda h, r: sc_tuned.score_all_tails(H_tuned, h, r)

    # 3. ComplEx
    if Path("results/rgcn_complex.pt").exists():
        ckpt_c = torch.load("results/rgcn_complex.pt", weights_only=False)
        nf_c = NodeFeatures(node_types, emb_dim=128)
        enc_c = RGCNEncoder(nf_c.out_dim, 256, 256, 12, 4, 0.2)
        dec_c = ComplExDecoder(6, 256)
        nf_c.load_state_dict(ckpt_c["node_features"])
        enc_c.load_state_dict(ckpt_c["encoder"])
        dec_c.load_state_dict(ckpt_c["decoder"])
        nf_c.eval(); enc_c.eval(); dec_c.eval()
        with torch.no_grad():
            H_c = enc_c(nf_c(), edge_index, edge_type)
        models_info["R-GCN+ComplEx"] = lambda h, r: dec_c.score_all_tails(H_c, h, r)

    # 4. RotatE
    if Path("results/rgcn_rotate.pt").exists():
        ckpt_r = torch.load("results/rgcn_rotate.pt", weights_only=False)
        nf_r = NodeFeatures(node_types, emb_dim=128)
        enc_r = RGCNEncoder(nf_r.out_dim, 256, 256, 12, 4, 0.2)
        dec_r = RotatEDecoder(6, 256, gamma=6.0)
        nf_r.load_state_dict(ckpt_r["node_features"])
        enc_r.load_state_dict(ckpt_r["encoder"])
        dec_r.load_state_dict(ckpt_r["decoder"])
        nf_r.eval(); enc_r.eval(); dec_r.eval()
        with torch.no_grad():
            H_r = enc_r(nf_r(), edge_index, edge_type)
        models_info["R-GCN+RotatE"] = lambda h, r: dec_r.score_all_tails(H_r, h, r, chunk_size=100)

    # 5. Full VRGAE
    if Path("results/vrgae_best.pt").exists():
        ckpt_v = torch.load("results/vrgae_best.pt", weights_only=False)
        cfg_v = {
            "encoder": {"emb_dim": 128, "hidden_dim": 256, "out_dim": 256, "num_relations": 12, "num_bases": 4, "dropout": 0.2},
            "decoder": {"num_base_relations": 6, "gamma": 6.0},
            "model": {"decoder": "rotate"},
        }
        vrgae = VRGAE(node_types, cfg_v)
        vrgae.load_state_dict(ckpt_v["model_state"])
        vrgae.eval()
        models_info["VRGAE (Full)"] = vrgae.make_score_all_tails(edge_index, edge_type)

    print("==========================================================================================")
    print("PHASE 4: DETAILED PER-RELATION BREAKDOWN ON VALIDATION SET (UNRESTRICTED FILTERED RANKING)")
    print("==========================================================================================")

    val_results = {}
    type_constrained_val_results = {}
    test_results = {}

    for name, scorer in models_info.items():
        print(f"\n--- Evaluating: {name} on Full Validation (1,942 triples) ---")
        val_res = evaluate_ranking(scorer, val_triples, known_all, num_entities, batch_size=256, relation_names=rel_names)
        val_results[name] = val_res

        # Auxiliary Type-constrained ranking
        tc_res = evaluate_type_constrained(scorer, val_triples, known_all, node_types, type_names, rel_names, rel_to_tail_type)
        type_constrained_val_results[name] = tc_res

        # Single-pass test evaluation
        print(f"--- Evaluating: {name} on Test Set (1,942 triples, single pass) ---")
        test_res = evaluate_ranking(scorer, test_triples, known_all, num_entities, batch_size=256, relation_names=rel_names)
        test_results[name] = test_res

    # Output Validation Table
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS SUMMARY (UNRESTRICTED FILTERED RANKING, N=11,417)")
    print("=" * 80)
    print(f"{'Model':<30} | {'MRR':<8} | {'Hits@1':<8} | {'Hits@3':<8} | {'Hits@10':<8} | {'Mean Rank':<10}")
    print("-" * 80)
    for name, res in val_results.items():
        print(f"{name:<30} | {res['mrr']:<8.4f} | {res['hits@1']:<8.4f} | {res['hits@3']:<8.4f} | {res['hits@10']:<8.4f} | {res['mean_rank']:<10.1f}")

    # Output ManifestAs Comparison Table
    print("\n" + "=" * 80)
    print("RELATION BREAKDOWN: ManifestAs (1,055 validation triples, 54.3% of val)")
    print("=" * 80)
    print(f"{'Model':<30} | {'MRR':<8} | {'Hits@1':<8} | {'Hits@3':<8} | {'Hits@10':<8} | {'Mean Rank':<10}")
    print("-" * 80)
    for name, res in val_results.items():
        m_rel = res["per_relation"].get(4, {})
        print(f"{name:<30} | {m_rel.get('mrr', 0.0):<8.4f} | {m_rel.get('hits@1', 0.0):<8.4f} | {m_rel.get('hits@3', 0.0):<8.4f} | {m_rel.get('hits@10', 0.0):<8.4f} | {m_rel.get('mean_rank', 0.0):<10.1f}")

    # Output Type-Constrained Auxiliary Table
    print("\n" + "=" * 80)
    print("AUXILIARY ABLATION: TYPE-CONSTRAINED VALIDATION RANKING")
    print("=" * 80)
    print(f"{'Model':<30} | {'MRR':<8} | {'Hits@1':<8} | {'Hits@3':<8} | {'Hits@10':<8} | {'Mean Rank':<10}")
    print("-" * 80)
    for name, res in type_constrained_val_results.items():
        print(f"{name:<30} | {res['mrr']:<8.4f} | {res['hits@1']:<8.4f} | {res['hits@3']:<8.4f} | {res['hits@10']:<8.4f} | {res['mean_rank']:<10.1f}")

    # Output Single-Pass Test Table
    print("\n" + "=" * 80)
    print("PHASE 5: SINGLE-PASS TEST RESULTS (TEST SET: 1,942 TRIPLES, COMPUTED ONCE)")
    print("=" * 80)
    print(f"{'Model':<30} | {'Test MRR':<9} | {'Hits@1':<8} | {'Hits@3':<8} | {'Hits@10':<8} | {'Mean Rank':<10}")
    print("-" * 80)
    for name, res in test_results.items():
        print(f"{name:<30} | {res['mrr']:<9.4f} | {res['hits@1']:<8.4f} | {res['hits@3']:<8.4f} | {res['hits@10']:<8.4f} | {res['mean_rank']:<10.1f}")


if __name__ == "__main__":
    main()
