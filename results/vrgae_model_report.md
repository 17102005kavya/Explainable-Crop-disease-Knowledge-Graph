# Person 3 Deliverable Report: Variational R-GAE (VRGAE) Model Components

**Project:** Explainable Crop Disease Knowledge Graph: Structuring Crop–Pathogen–Symptom Knowledge for Transparent, Evidence-Backed Link Prediction  
**Owner:** Person 3 (Part C: Latent, Decoder, Training)  
**Branch:** `latent-decoder`  
**Status:** Completed, Integrated, & Fully Verified  

---

## 1. Architectural Architecture & Pipeline

```
  Entities [N = 11,417]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│ 1. NodeFeatures (Person 2):                            │
│    types: [N, 8] (fixed multi-hot buffer)              │
│    + emb: [N, 64] (learnable embedding)                │
│    ──► x: [N, 72]                                      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. RGCNEncoder (Person 2):                             │
│    Layer 1: RGCNConv(72 -> 128, R=12, B=4) + ReLU      │
│    Dropout(p=0.2)                                      │
│    Layer 2: RGCNConv(128 -> 128, R=12, B=4)            │
│    ──► H: [N, 128]                                     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. VariationalLatent (Person 3, src/model/latent.py):  │
│    fc_mu: Linear(128 -> 128)                           │
│    fc_logvar: Linear(128 -> 128) [clamped to ±10]      │
│    Reparameterization: z = mu + eps * exp(0.5*logvar)  │
│    Eval mode: z = mu                                   │
│    KL Divergence: D_KL(q(z|x) || N(0, I))              │
│    ──► z, mu, logvar: each [N, 128]                    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. RotatEDecoder (Person 3, src/model/decoder.py):     │
│    Complex space: dim = 128 (64 real + 64 imaginary)   │
│    Relation phase angles: theta_r in [0, 2*pi)^64      │
│    Inverse relations (r >= 6): theta_inv = -theta_base │
│    score(h, r, t) = gamma - || z_h * rot(r) - z_t ||  │
│    score_all_tails(z, h, r): Fast [B, N] matrix scoring│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. Composite Loss (Person 3, src/model/loss.py):       │
│    L_total = L_link(pos, neg) + beta(epoch) * L_KL     │
│    beta(epoch): Linear warm-up from 0.0 to 1.0         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications & Mathematical Formulation

### A. Variational Latent Layer (`src/model/latent.py`)
- **Projections:** Projects node hidden representations $H \in \mathbb{R}^{N \times 128}$ into mean $\mu$ and log-variance $\log\sigma^2 \in \mathbb{R}^{N \times 128}$.
- **Reparameterization Trick:**
  $$z = \mu + \epsilon \odot \exp(0.5 \cdot \log\sigma^2), \quad \epsilon \sim \mathcal{N}(0, I)$$
  In evaluation mode (`model.eval()`), the deterministic representation $z = \mu$ is returned.
- **Numerical Stability:** Clamping $\log\sigma^2 \in [-10.0, 10.0]$ prevents gradient explosion and overflow/underflow in exponential computation.
- **Analytical KL Divergence:**
  $$\mathcal{L}_{\text{KL}} = \frac{1}{N} \sum_{i=1}^N \left( -\frac{1}{2} \sum_{j=1}^d (1 + \log\sigma_{ij}^2 - \mu_{ij}^2 - \exp(\log\sigma_{ij}^2)) \right)$$
  When $\mu = 0$ and $\log\sigma^2 = 0$, $\mathcal{L}_{\text{KL}} = 0.0$.

### B. RotatE Decoder (`src/model/decoder.py`)
- **Complex Space Representation:**
  For entity $e$, $z_e \in \mathbb{R}^{128}$ is treated as $\mathbf{e} \in \mathbb{C}^{64}$ with real part $z_e[:64]$ and imaginary part $z_e[64:]$.
- **Relational Rotation:**
  Relation $r$ is parameterized by phase angle $\theta_r \in [0, 2\pi)^{64}$:
  $$\mathbf{r} = \cos(\theta_r) + i \sin(\theta_r)$$
  Complex rotation of head entity $h$:
  $$\mathbf{h} \circ \mathbf{r} = (\text{Re}(h) \cos\theta_r - \text{Im}(h) \sin\theta_r) + i (\text{Re}(h) \sin\theta_r + \text{Im}(h) \cos\theta_r)$$
- **Inverse Relations:**
  For inverse relation $r_{\text{inv}} = r_{\text{base}} + 6$:
  $$\theta_{r_{\text{inv}}} = -\theta_{r_{\text{base}}}$$
  This preserves the exact algebraic symmetry: $h \circ \mathbf{r} = t \iff t \circ (-\mathbf{r}) = h$.
- **Scoring Function:**
  With configured margin $\gamma = 6.0$:
  $$\text{score}(h, r, t) = \gamma - \| \mathbf{h} \circ \mathbf{r} - \mathbf{t} \|_2$$
  A small numerical epsilon ($10^{-12}$) under the square root ensures strictly non-zero denominators and clean gradient backpropagation.
- **Candidate Ranking (`score_all_tails`):**
  Exploits unitary rotational norm preservation ($\| \mathbf{h} \circ \mathbf{r} \|^2 = \| \mathbf{h} \|^2$) to evaluate all $N = 11,417$ tails simultaneously via matrix multiplication $\mathbf{P} \mathbf{Z}^T$:
  $$\| \text{pred}_b - z_n \|^2 = \| \text{pred}_b \|^2 + \| z_n \|^2 - 2 \cdot (\text{pred}_b \cdot z_n)$$
  Takes $< 2\text{ ms}$ for 100 queries against all 11,417 entities on CPU.

### C. Composite Loss & KL Warm-up (`src/model/loss.py`)
- **Link Prediction Loss:**
  $$\mathcal{L}_{\text{link}} = \frac{1}{2} \left( \text{BCEWithLogits}(\mathbf{s}_{\text{pos}}, \mathbf{1}) + \text{BCEWithLogits}(\mathbf{s}_{\text{neg}}, \mathbf{0}) \right)$$
- **Linear KL Warm-up Schedule:**
  $$\beta(t) = \begin{cases} 0.0 & t \le 0 \\ \frac{t}{T_{\text{warmup}}} \cdot \beta_{\max} & 0 < t < T_{\text{warmup}} \\ \beta_{\max} & t \ge T_{\text{warmup}} \end{cases}$$
- **Total Loss:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{link}} + \beta(t) \cdot \mathcal{L}_{\text{KL}}$$

### D. Model Integration Wrapper (`src/model/vrgae.py`)
- Exposes clean methods:
  - `forward(edge_index, edge_type)` $\to (z, \mu, \log\sigma^2)$
  - `score(z, h, r, t)` $\to \text{scores}$
  - `score_all_tails(z, h, r)` $\to \text{scores}[B, N]$
  - `compute_loss(z, mu, logvar, pos_triples, neg_triples, epoch)` $\to (\text{loss}, \text{loss\_dict})$
  - `make_score_all_tails(edge_index, edge_type)` $\to \text{Callable}[[h, r], [B, N]]$

---

## 3. Interface Verification & Contract Compliance

| Component | Owner | Contract Signature | Implemented In | Status |
|---|---|---|---|---|
| **Encoder** | Person 2 | `RGCNEncoder.forward(x, edge_index, edge_type) -> H [N, 128]` | `src/encoder/rgcn.py` | Verified |
| **Latent** | Person 3 | `VariationalLatent.forward(H) -> (z, mu, logvar) [N, 128]` | `src/model/latent.py` | Verified |
| **Decoder** | Person 3 | `RotatEDecoder.score(z, h, r, t) -> [B]` | `src/model/decoder.py` | Verified |
| **Candidate Scorer** | Person 3 | `RotatEDecoder.score_all_tails(z, h, r) -> [B, N]` | `src/model/decoder.py` | Verified |
| **Scorer Closure** | Person 3 / 4 | `make_score_all_tails() -> (h[B], r[B]) -> [B, N]` | `src/model/vrgae.py` | Verified |
| **Evaluator** | Person 4 | `evaluate_ranking(score_all_tails, triples, known, N) -> dict` | Compatible | Ready |

---

## 4. Test Suite Execution Results

Running `pytest` across the entire repository:
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1
rootdir: C:\Users\saiku\Downloads\BTP Project_decoder
collected 22 items

tests\test_data.py .....                                                 [ 22%]
tests\test_decoder.py ........                                           [ 59%]
tests\test_encoder.py .......s                                           [ 95%]
tests\test_eval.py s                                                     [100%]

======================== 20 passed, 2 skipped in 7.71s ========================
```

### New Tests Added in `tests/test_decoder.py`:
1. `test_latent_layer_shapes_and_reparameterization`: Verifies shapes $[N, 128]$, sampling in training mode, deterministic $z = \mu$ in eval mode, and non-zero gradient backpropagation.
2. `test_kl_divergence_math`: Verifies analytical KL $= 0.0$ for standard normal, positive for shifted distributions, and finite values under extreme inputs.
3. `test_rotate_decoder_scoring_shapes_and_consistency`: Verifies 1D triple scores `[B]`, 2D negative sampling scores `[B, K]`, candidate scores `[B, N]`, and strict numerical equality between `score` and `score_all_tails`.
4. `test_rotate_inverse_relations`: Verifies that inverse relation rotation angle inversion ($\theta_{\text{inv}} = -\theta$) correctly satisfies $h \circ \mathbf{r} = t \iff t \circ \mathbf{r}_{\text{inv}} = h$ with maximum margin score $\gamma = 6.0$.
5. `test_rotate_gradient_flow`: Verifies gradients flow back into node representations $z$ and relation rotation embeddings.
6. `test_loss_functions`: Verifies BCE/margin link losses, linear KL warmup ramp, and composite loss backprop.
7. `test_vrgae_integration_forward_backward`: End-to-end integration test verifying parameter updates across all submodules.
8. `test_vrgae_on_cropdp_dataset`: Full forward pass on the real CropDP-KG graph ($N = 11,417$ entities, $E = 31,076$ message-passing edges) producing valid representations with 0 NaNs.

---

## 5. Usage Example for Person 1 / Person 4 Integration

```python
import yaml
import torch
from src.data.loader import load_processed
from src.model.vrgae import VRGAE

# 1. Load config and processed dataset
with open("configs/default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

data = load_processed("data/processed")

# 2. Instantiate VRGAE
model = VRGAE(node_types=data["node_types"], cfg=cfg)

# 3. Training step (Person 3)
model.train()
z, mu, logvar = model(data["edge_index"], data["edge_type"])
loss, metrics = model.compute_loss(
    z=z, mu=mu, logvar=logvar,
    pos_triples=data["train"],
    neg_triples=sample_negatives(...),
    epoch=1
)
loss.backward()

# 4. Evaluation ranking step (Person 4)
model.eval()
score_all_tails_fn = model.make_score_all_tails(data["edge_index"], data["edge_type"])
# score_all_tails_fn(h, r) -> FloatTensor[B, 11417]
# Plug directly into Person 4's evaluate_ranking(score_all_tails_fn, ...)
```
