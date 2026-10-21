# CropDP-KG — Current Work Status and Next Actions

**Review date:** 2026-10-20
**Basis:** current source, configs, test definitions, README, and recorded `results/results.csv`; no training or data generation performed for this update.

## Status definitions

- **Implemented in source:** corresponding code exists.
- **Recorded:** repository result rows or documentation state that a run occurred.
- **Verified in this review:** directly checked in the current checkout or executed during the review.
- **Unverified:** supporting data, checkpoint, log, or successful test output is not available.

Do not treat “implemented” or “recorded” as equivalent to “tested” or “reproduced.”

## Person 1 — Data and knowledge graph preparation

### Implemented in source

- Multi-encoding CSV loading and relation-column/file-name handling.
- Unicode normalization, missing-value, self-loop, and exact-duplicate detection.
- Near-duplicate entity detection; entity merging is opt-in.
- Conflict reporting for selected functional/type patterns.
- Entity/relation ID creation, multi-hot types, entity-covering splits, and train-only graph edges with inverse edges.
- Uniform head/tail corruption sampler with known-positive filtering, a retry limit, and optional primary-type constraints.
- Generation of provenance, conflict, metadata, and data-report files.
- Data sanity tests in `tests/test_data.py`.

### Not verified in this checkout

- No raw or processed dataset is present; actual counts, IDs, conflicts, duplicate removals, split properties, and report contents cannot be confirmed.
- `pytest` was unavailable, so the data tests were not run.
- The code records reversed-pair leakage but does not fail when it is nonzero.
- The sampler can return unresolved known-positive collisions after its retry limit; its `last_unresolved` value is not made a hard error.

### Person 1 action list (priority order)

1. **Supply/document source data.** Put the authorized CropDP-KG CSVs in the documented raw-data location or provide reproducible acquisition steps and source/license attribution.
2. **Regenerate processed data with the intended relation set.** From the repository root, run:
   ```powershell
   python src\data\prepare_cropdp.py --raw data\raw --out data\processed --seed 42 --drop-relations EnglishName
   ```
3. **Validate all outputs.** Check `meta.json`, entity/relation tables, split arrays, type features, train-only graph, fixed negatives, `provenance.csv`, `conflicts.csv`, and `data_report.md`.
4. **Review counts and flagged records.** Report actual counts from generated files; inspect conflicts and near duplicates rather than assuming they were resolved.
5. **Close leakage/sampler checks.** Decide whether reverse ordered pairs are disallowed by protocol and enforce/report that decision; fail or explicitly report if negative sampling ends with unresolved collisions.
6. **Run `tests/test_data.py` in the supported environment** and retain actual output.

### Acceptance criteria

- A teammate can reproduce the documented split and mappings from the sourced raw files using the declared seed/settings.
- Report counts match the generated files and include per-relation split counts, conflicts/drops, graph properties, and leakage checks.
- Data tests pass on the generated dataset; negative arrays contain no known positives; unresolved sampler collisions are zero or explicitly handled.

## Person 2 — R-GCN encoder

### Implemented / recorded

- Type features plus learnable entity embeddings; two-layer basis-decomposed PyG R-GCN.
- Standalone DistMult training and a staged tuning script.
- Multiple tuning/seed records in `results.csv`.

### Remaining

- Dataset-backed tests, training logs, and checkpoints are absent.
- The selected P2 tuning values described in CSV notes do not match `configs/rgcn_distmult.yaml`.
- The P4 R-GCN baseline uses a separate `RGCNEncoderLite` implementation.
- Standalone validation ranking does not average ties like the shared evaluator.

### Acceptance criteria

- Run targeted tests and a recorded encoder training smoke test on the restored data.
- Align the final baseline config with the selected run and retain its checkpoint/log.
- Document encoder differences and ensure final test ranking uses the shared evaluator.

## Person 3 — Latent variables, decoder, and VRGAE training

### Implemented / recorded

- Current `src/model/latent.py` and `src/model/loss.py` contain implementations; earlier statements that they are stubs are stale.
- RotatE/ComplEx decoder and integrated VRGAE source are present.
- CSV records include multiple VRGAE outcomes.

### Remaining

- No VRGAE checkpoint or training log is present in this checkout.
- Several VRGAE training paths use different batching, masking, negative sampling, optimizers, and evaluation.
- The cause of the large metric discrepancies is not established.
- No dedicated training path was identified for the project RotatE decoder independently from VRGAE.

### Acceptance criteria

- Select and document one training protocol for headline results.
- Preserve config, code revision, split/mapping identifiers, checkpoint, validation epoch, and loss log for every reported run.
- Execute latent, decoder, loss, integration, and short training tests.
- Evaluate all reported checkpoints with the same filtered evaluator and explain the distinct CSV results.

## Person 4 — Evaluation and baselines

### Implemented / recorded

- Shared filtered tail-ranking/AUC evaluator, baseline implementations, YAML runner, CSV logging, and table generator exist.
- The displayed final table agrees with its recorded CSV rows.
- New runner output records source/config/data/environment fingerprints, protocol, selected epoch, and saved checkpoint checksums; `reproduce.py` verifies or reruns fully-provenanced rows.

### Remaining

- Underlying data, checkpoints, and logs are absent, so results cannot be independently reproduced.
- Historical CSV rows are marked `legacy-unverified`; their provenance and artifacts cannot be retroactively recovered from this checkout.
- Baseline and VRGAE training paths do not all share negative generation.
- Baseline config does not reflect the P2 tuning winner.
- Macro averages and per-relation counts are represented in the code/CSV, but raw model outputs are absent.

### Acceptance criteria

- Run each model from a documented config and the same declared data split/evaluation protocol.
- Regenerate `results.csv` and the final table from retained run outputs.
- Keep test metrics as one-time held-out evaluation and report single-seed limitations.

## Cross-cutting completion checklist

- [ ] Raw data provenance and processed data available or reproducibly obtainable.
- [ ] Generated preprocessing report and tables agree with actual arrays.
- [ ] Leakage and sampler checks are explicit and tested.
- [ ] Supported dependencies/environment documented; tests executed and output retained.
- [ ] One authoritative training/evaluation path selected for comparisons.
- [ ] Checkpoints and logs map to every claimed result.
- [ ] All four reports describe the current source accurately.
- [ ] Historical metrics are reconciled with the PDF and regenerated from retained run artifacts.

## Overall verdict

The repository is **partially implemented**. The existing code and CSV demonstrate substantial work, but the project cannot honestly claim verified, reproducible results until the data, checkpoints, logs, and test execution are restored and reconciled.
