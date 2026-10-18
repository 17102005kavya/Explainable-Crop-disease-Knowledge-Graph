# Person 2 Baseline Report: R-GCN Encoder + DistMult Scorer

**Date:** 2026-10-18  
**Owner:** Person 2 (Part B)  
**Dataset:** CropDP-KG (`data/processed`)  
**Status:** Completed & Verified  

---

## 1. Architecture Overview

```
Entities [11,417]
       │
       ▼
┌──────────────────────────────────────────────┐
│ NodeFeatures:                                │
│   types [11417, 8] (fixed multi-hot buffer)  │
│   + emb [11417, 64] (learnable embedding)    │
│   ──► Output x: [11417, 72]                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ RGCNEncoder:                                 │
│   Layer 1: RGCNConv(72 -> 128, R=12, B=4)    │
│   Activation: ReLU + Dropout(p=0.2)          │
│   Layer 2: RGCNConv(128 -> 128, R=12, B=4)   │
│   ──► Output H: [11417, 128]                 │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ DistMultScorer (Standalone Baseline Scorer): │
│   Relations: 6 base relation embeddings      │
│   dim: 128 (Xavier uniform init)             │
│   score(h, r, t) = <H[h] * R[r], H[t]>       │
└──────────────────────────────────────────────┘
```

* **NodeFeatures:** Combines an 8-dimensional fixed multi-hot entity type vector with a 64-dimensional learnable embedding ($d_{in} = 72$).
* **RGCNEncoder:** Two `torch_geometric.nn.RGCNConv` layers with $B=4$ basis decomposition matrices and $R=12$ relations (6 forward base relations + 6 inverse relations). Intermediate ReLU activation and $0.2$ dropout. Output representation $H \in \mathbb{R}^{11417 \times 128}$.
* **DistMultScorer:** Temporary baseline scorer using 6 forward relation embeddings ($d = 128$) to evaluate representation quality before integration with Person 3's Variational Latent and RotatE Decoder.

---

## 2. Training Procedure

* **Configuration:** `configs/default.yaml`
  * Total Epochs: 50
  * Optimizer: Adam ($\text{lr} = 0.001$)
  * Loss Function: Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`)
  * Negatives per Positive ($k$): 16
  * Evaluation Interval: Every 10 epochs
* **Message-Passing Graph:**
  * Uses only the train-only graph (`graph_edge_index.npy`, `graph_edge_type.npy`) with 31,076 directed edges (15,538 forward + 15,538 inverse).
  * No validation or test edges are included in message passing.
* **Negative Sampling:**
  * `UniformNegativeSampler` corrupts either head or tail uniformly.
  * Negatives are filtered against all 19,422 known triples across `train`, `val`, and `test` splits (0% false negative contamination).

---

## 3. Evaluation Methodology

* **Filtered Tail Prediction:** For each triple $(h, r, t)$ in validation:
  1. Score all $N = 11,417$ candidate entities using `scorer.score_all_tails(H, h, r)`.
  2. Filter out all other known ground-truth tails $(h, r, t') \in \text{known}$ where $t' \ne t$ by masking their scores to $-10^9$. The true target tail $t$ is strictly preserved.
  3. Compute filtered rank: $1 + \sum_{t'} \mathbb{I}(s_{t'} > s_t)$.
  4. Metrics:
     * **MRR:** Mean Reciprocal Rank ($\frac{1}{\text{rank}}$)
     * **Hits@1:** Fraction of triples ranked at position 1
     * **Hits@3:** Fraction of triples ranked $\le 3$
     * **Hits@10:** Fraction of triples ranked $\le 10$

---

## 4. Benchmark Results

### 50-Epoch Training Convergence (500-Triple Intermediate Slice)

| Epoch | Loss | Filtered MRR | Hits@1 | Hits@3 | Hits@10 | Epoch Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.2980 | 0.0034 | 0.0000 | 0.0000 | 0.0020 | 0.79s |
| 20 | 0.2159 | 0.0151 | 0.0000 | 0.0040 | 0.0340 | 0.78s |
| 30 | 0.1767 | 0.0226 | 0.0000 | 0.0040 | 0.0560 | 0.82s |
| 40 | 0.1546 | 0.0277 | 0.0000 | 0.0040 | 0.0600 | 0.80s |
| 50 | 0.1351 | 0.0309 | 0.0000 | 0.0060 | 0.0720 | 0.74s |

### Full Validation Benchmark (All 1,942 Validation Triples)

| Evaluation Scope | Evaluated Triples | Filtered MRR | Hits@1 | Hits@3 | Hits@10 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Intermediate Slice** | 500 | 0.0309 | 0.0000 | 0.0060 | 0.0720 |
| **Full Validation Set** | **1,942** | **0.0121** | **0.0000** | **0.0021** | **0.0242** |

---

## 5. Saved Artifacts

* **Best Model Checkpoint (50 epochs):** `results/encoder_checkpoint.pt`
* **Full-Validation Evaluated Checkpoint:** `results/encoder_checkpoint_full_val.pt`
* **Training Log:** `results/encoder_train.log`
* **Experiment Log:** `results/results.csv`
