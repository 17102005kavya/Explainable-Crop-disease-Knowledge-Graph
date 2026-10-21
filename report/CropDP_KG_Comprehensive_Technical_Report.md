# CropDP-KG VRGAE Project — Evidence-Based Repository Report

**Review date:** 2026-10-20
**Branch at review:** `main`
**Scope:** Read-only review of source, configs, tests, README, and tracked result records. No training was launched.

## 1. Executive summary

The repository contains implementations for preprocessing, R-GCN encoding, variational latents, decoders, training, baselines, and evaluation. However, implementation presence is not proof of successful execution. The checkout reviewed did not contain raw or processed data, generated data reports, model checkpoints, or training logs. The test command could not run because `pytest` was not installed.

The prior audit matrix had 30 requirement entries: **2 Complete, 27 Partial, 1 Not implemented**. Complete here means code and supporting repository evidence were present; it does not mean this review independently reran the experiment. Most entries remain Partial because execution and artifact evidence are unavailable.

The key blockers are:

1. **Missing data and run artifacts:** metrics in `results/results.csv` cannot be tied to inspected checkpoints, split arrays, or logs.
2. **Multiple, non-equivalent training paths:** negative sampling, batch edge masking, optimizers, validation limits, and loss handling differ between scripts.
3. **VRGAE result discrepancy:** the CSV contains more than one shared-recipe result and a much lower checkpoint result; existing evidence does not establish a single cause.

### Provenance follow-up

The experiment runner now records protocol, Git commit/dirty state, a source-code
fingerprint, the original and effective config, a processed-data fingerprint,
environment versions, selected epoch, and saved-checkpoint checksums for new runs.
`reproduce.py` validates those identities and can rerun a selected run against its
recorded metrics. The existing rows predate this instrumentation and are marked
`legacy-unverified`; their provenance cannot be reconstructed from the current
checkout. The processed dataset and historical checkpoints are still absent, so the
new tooling does not independently reproduce the existing results.

The variational modules are **implemented in the current source**. Earlier statements in the reports that `latent.py` and `loss.py` were stubs are stale.

## 2. Repository and evidence state

- Git branch: `main`; the worktree contained an untracked `report/` directory at review time. Those report files were user data and were not changed during the audit.
- `data/` contained only `README.md`. There were no `raw/` or `processed/` datasets.
- `results/` contained reports, `results.csv`, and `final_table.md`; no model checkpoints or training logs were present.
- Python source syntax check: 29 Python files parsed with no syntax errors.
- Test execution was attempted with `python -m pytest -p no:cacheprovider -q`; it stopped before collection because `pytest` was unavailable.
- No training or data generation was run.

The claimed data counts in `README.md` (11,417 entities, 6 relations, 19,422 triples, split sizes 15,538/1,942/1,942) are **documentation claims**, not verified dataset statistics in this checkout.

## 3. Data preparation — Person 1

`src/data/prepare_cropdp.py` implements the main preparation pipeline:

- `load_raw` discovers `relation*.csv`, supports multiple encodings, and chooses columns or filename relation hints.
- `norm` applies Unicode NFKC normalization, removes selected zero-width characters, and normalizes whitespace.
- `clean` marks missing values, self-loops, and exact duplicates for dropping. Near-duplicate entities are detected; merging them requires `--merge-near-dups`.
- `find_conflicts` reports several conflict types to `conflicts.csv`; it flags these for review rather than resolving their meaning.
- `build_ids` creates sorted entity IDs and relation IDs, then produces the entity type matrix.
- `make_splits` uses seeded per-relation selection and attempts entity coverage. Repeated ordered `(head, tail)` pairs under different relations are retained in train or moved there.
- The processing pipeline builds message-passing edges from train triples only and adds inverse edges.
- `UniformNegativeSampler` corrupts a head or tail and filters known positives during retries. It exposes `last_unresolved`, but after `max_tries` it returns any remaining unresolved samples rather than failing.
- `write_report` can generate counts, relation breakdown, graph properties, flagged conflicts, and leakage-check output.

**Not verified:** actual entity/relation/triple counts, duplicate/conflict counts, split overlap, ID mappings, report accuracy, or negative collisions. The generated data artifacts are absent.

**Leakage note:** the code reports reversed `(tail, head)` pairs in train for validation/test but does not assert they are absent. It prevents exact-triple overlap and has logic for repeated ordered pairs; reverse-edge leakage should be explicitly reviewed against the intended protocol.

## 4. Encoder — Person 2

- `src/encoder/node_features.py`: fixed type-feature buffer concatenated with a trainable entity embedding.
- `src/encoder/rgcn.py`: two `torch_geometric.nn.RGCNConv` layers, relation-specific weights with basis decomposition, ReLU and dropout between layers.
- `src/encoder/train_encoder.py`: standalone DistMult-scored training path with BCE, validation ranking, and best-validation-MRR checkpoint saving.
- `src/encoder/tune_baseline.py`: staged baseline tuning code; `results.csv` includes multiple recorded settings and seeds.

The independent result records are not accompanied by corresponding logs or checkpoints in this checkout. The final P4 configuration in `configs/rgcn_distmult.yaml` also differs from the P2 tuning winner described in CSV notes. Additionally, P4's `RGCNEncoderLite` is a separate encoder implementation, so its baseline results should not be described as the same encoder implementation used by VRGAE.

`train_encoder.py` uses a custom ranking implementation that ranks ties optimistically, whereas the shared evaluator averages ties. That can affect checkpoint selection, even though final P4 rankings use the shared evaluator.

## 5. Latent, decoders, and VRGAE — Person 3

- `src/model/latent.py` defines `VariationalLatent` with `mu` and `logvar` linear heads, clamped log-variance, stochastic reparameterization while training, and deterministic `z = mu` during evaluation. KL is summed over latent dimensions and averaged over nodes.
- `src/model/loss.py` implements BCE-style softplus or margin link loss, linear KL warm-up, and a combined loss breakdown.
- `src/model/decoder.py` contains RotatE and ComplEx scorers. RotatE scores `gamma - distance`, handles inverse relation IDs by negating the relation angle, and supports tail ranking.
- `src/model/vrgae.py` wires node features, R-GCN, latent layer, and decoder.
- `src/model/train.py`, `src/model/train_models.py`, and `src/eval/vrgae_adapter.py` each provide distinct VRGAE-related training/evaluation paths.

The model and unit/integration test code exist. This review did not execute those tests or inspect a resulting checkpoint. A standalone training procedure for the project's `RotatEDecoder` class was not identified; the separate plain RotatE baseline is not the same decoder implementation.

## 6. Evaluation and baselines — Person 4

`src/eval/evaluator.py` provides filtered tail ranking, average tie ranks, micro metrics, per-relation metrics, unweighted macro metrics, and AUC. `src/eval/run_experiment.py` calls the shared evaluator for test ranking and uses fixed negatives for AUC. `src/eval/make_table.py` assembles the table from CSV run groups.

The repository has code for popularity, plain RotatE, R-GCN+DistMult, and a no-graph DistMult ablation. Their training procedures are not all uniform: the plain RotatE and R-GCN baseline fit methods use direct random corruptions; the shared VRGAE adapter also generates random corruptions directly, while other paths use the filtered `UniformNegativeSampler`.

The final-table CSV values are consistent with the values displayed in `results/final_table.md`; this confirms table-to-CSV consistency only, not the underlying model evaluations. The CSV records per-relation test counts of 156, 203, 387, 132, 1,055, and 9 (1,942 total). TemperIs has very few test examples.

## 7. Performance records and VRGAE discrepancy

The repository records these outcomes:

| Result record | Test MRR | Hits@10 | AUC | Provenance available here |
|---|---:|---:|---:|---|
| RotatE alone | 0.1963 | 0.3414 | 0.8418 | CSV row group and config; no checkpoint/log |
| VRGAE shared recipe, earlier run | 0.0072 | 0.0165 | 0.5688 | CSV row group; best epoch 2 noted |
| VRGAE shared recipe, later run | 0.0834 | 0.1627 | 0.9057 | CSV row group; best epoch 30 noted |
| VRGAE checkpoint evaluation | 0.0045 | 0.0093 | 0.4782 | CSV references `results/vrgae_checkpoint.pt`, but file is absent |
| Person 3 training record | 0.0259 | 0.0772 | Not recorded | CSV record; no matching checkpoint/log present |

The source confirms differences between paths: full-batch versus mini-batch training, batch-edge masking versus full train graph, filtered sampler versus unfiltered direct corruption, Adam versus AdamW (and in one path cosine scheduling), and differing evaluation limits/evaluators. These are plausible comparability issues, but **the cause of the specific result discrepancy is unverified**. The checkpoint, data, exact mapping tables, and loss logs are needed to determine whether dimensions, weights, training behavior, or evaluation explain it.

Higher scores are used for more plausible triples: RotatE returns `gamma - distance`; the evaluator ranks greater scores ahead of the target. The AUC warning is justified by code: default negatives are sampled from all entities without relation-specific type constraints. The proportion of wrong-type negatives cannot be computed without the absent arrays.

## 8. Reproducibility verdict

The project has substantial implementation coverage but is **not reproducible from the reviewed checkout**. Before treating reported metrics as verified, restore the raw/processed dataset and the exact model checkpoints and logs; reconcile the competing training paths; run the tests; then regenerate the evaluation records and table from the declared artifacts.

For the owner-by-owner next steps, see `CropDP_KG_Completed_vs_Pending_Work.md`. For commands and test instructions, see `CropDP_KG_Test_and_Reproduction_Guide.md`.
