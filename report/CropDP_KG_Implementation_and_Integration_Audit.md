# CropDP-KG Implementation and Integration Audit

**Review date:** 2026-10-20
**Scope:** static source/config/test inspection and repository artifact inventory. This is not a successful test or training report.

## 1. Current artifact state

| Item | State in reviewed checkout |
|---|---|
| Branch | `main` |
| Raw dataset | Not present |
| Processed dataset | Not present; `data/` contains `README.md` only |
| Model checkpoints | Not present |
| Training logs | Not present |
| Result CSV and final table | Present |
| Tests | Present; not executed because `pytest` is unavailable |
| Python syntax check | 29 files parsed; no syntax errors |

## 2. Interfaces and model wiring

| Component | Current code evidence | Static conclusion |
|---|---|---|
| Node feature module | `src/encoder/node_features.py`, `NodeFeatures` | Fixed type matrix is concatenated with learned entity embeddings. |
| R-GCN | `src/encoder/rgcn.py`, `RGCNEncoder.forward` | Two relation-aware `RGCNConv` layers with basis decomposition; dropout/ReLU between layers. |
| Variational latent | `src/model/latent.py`, `VariationalLatent` | Mean/log-variance heads, clamped log variance, train-time reparameterization, eval-time mean. |
| KL loss | `VariationalLatent.kl_divergence`; `src/model/loss.py` | Standard diagonal-Gaussian KL to unit normal; sum over latent dimension then mean over nodes. |
| RotatE | `src/model/decoder.py`, `RotatEDecoder` | Complex rotation; score is `gamma - distance`; inverse relation uses negated angle. |
| VRGAE | `src/model/vrgae.py`, `VRGAE` | Composes node features, encoder, latent module, and decoder. |
| Shared scorer | `src/eval/vrgae_adapter.py`, `VRGAEScorer` | Uses deterministic `z = mu` for inference and exposes all-tail and triple scoring. |
| Ranking | `src/eval/evaluator.py`, `evaluate_ranking` | Filters other known tails; uses average rank for ties; returns overall, per-relation, and unweighted macro metrics. |

These are source-level conclusions. Shapes, gradients, device compatibility, and checkpoint loading were not executed in this review.

## 3. Data and leakage checks

`src/data/prepare_cropdp.py` builds IDs, splits, train-only graph edges, inverse edges, fixed negatives, and a report. `tests/test_data.py` defines checks for shapes/ranges, exact split overlap, entity coverage, graph edges, and fixed negatives.

Static findings:

- Entity and relation IDs are built once from cleaned triples, then used for the resulting splits.
- Same-triple split overlap is checked; entity coverage is enforced by the split logic.
- Repeated ordered `(head, tail)` pairs across relations are forced into train or moved there.
- Reverse ordered-pair leakage is counted by `leakage_check`, but is not asserted to be zero.
- Conflict detection reports candidates; it does not resolve their factual correctness.
- Near-duplicate merging is opt-in (`--merge-near-dups`).
- `UniformNegativeSampler` retries known-positive collisions up to `max_tries`; unresolved slots are counted but returned.
- No generated dataset or report is present to verify actual values.

## 4. Training/evaluation path differences

| Path | Sampling | Message-passing edges | Selection/evaluation |
|---|---|---|---|
| `src/encoder/train_encoder.py` | `UniformNegativeSampler` | Full train graph | Shared evaluator; full validation selection in protocol v2. |
| `src/model/train.py` | `UniformNegativeSampler` | Full train graph | Legacy full-batch path; not the selected comparison recipe. |
| `src/model/train_models.py` | `UniformNegativeSampler` | Full train graph | Separate full-batch path; not the selected comparison recipe. |
| `src/eval/vrgae_adapter.py` | `UniformNegativeSampler` | Hides current batch edges | Shared evaluator on full validation split; AdamW. |
| `src/eval/baselines.py` RotatE/R-GCN+DistMult | `UniformNegativeSampler` | R-GCN baseline masks current batch edges; plain RotatE has no graph | Shared evaluator and full validation model selection. |
| `src/eval/run_experiment.py` | Delegates to model implementation | Model-dependent | Shared evaluator for final test ranking; fixed negatives for AUC. |

Protocol v2 is the selected main comparison because it standardizes the negative
sampler, ranking evaluator, tie handling, and validation selection for the runner
models. The separately reported Person 3 full-batch checkpoint remains non-like-for-like.
The P4 baseline's custom `RGCNEncoderLite` is also distinct from the PyG R-GCN used by VRGAE/P2.

## 5. Results integrity

The selected protocol-v2 overall MRRs recorded in `results/results.csv` are: RotatE
`0.2134`, R-GCN + DistMult `0.1158`, DistMult without graph `0.1635`, and VRGAE-lite
shared recipe `0.0793`. The v2 rows state protocol, source commit, PyTorch build,
selected epoch, and config. All six v2 run groups are marked
`recorded-unverified` because full environment/config/data fingerprints, and most
checkpoints, are not retained. The 78 pre-v2 rows are marked `legacy-unverified`.
`make_table.py` prefers v2 rows and the generated `final_table.md` matches them; this
confirms recorded-table consistency, not independent result reproduction.

The AUC caveat is supported: default fixed negatives are corrupted from the unrestricted entity set, not constrained to the relation's expected tail type. No arrays are present to calculate how many negatives have an incorrect type.

New runs now carry explicit provenance fields and a saved checkpoint checksum. V2
rows retain the provenance actually stated in their notes; missing values are not
inferred. `reproduce.py` refuses to claim a rerun unless full provenance and artifacts
are available.

## 6. Previous report errors corrected

- `latent.py` and `loss.py` are **not stubs** in the current checkout.
- No claim that tests passed is made: the test suite was not run in this review.
- README dataset counts are labelled as documented claims, not verified artifact counts.
- Result rows are labelled recorded, not independently reproduced.
- Reports no longer claim checkpoints, data, or run logs exist in this checkout.

## 7. Required follow-up checks

1. Restore raw/processed data and run `tests/test_data.py`.
2. Inspect generated counts, conflicts, split leakage values, and unresolved negative samples.
3. Run targeted encoder, decoder, evaluator, and VRGAE tests in the documented PyTorch/PyG environment.
4. Restore checkpoints/logs or rerun the declared experiments, recording code revision, config, data/mapping identity, and best epoch.
5. Evaluate all claimed comparisons through one documented protocol and regenerate results.

See `CropDP_KG_Completed_vs_Pending_Work.md` for owner actions and acceptance criteria.
