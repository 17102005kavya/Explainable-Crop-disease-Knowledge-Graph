# Reproducing Part D (evaluator, baselines, VRGAE rows)

Everything in `results/final_table.md` comes from `results/results.csv`, and every row of the table comes from one
config in `configs/`. Rows written by the older team scripts (relation `all`, no `overall` row) are legacy rows and are
ignored by `make_table.py`.

## 1. Environment
Install the pinned environment with:

    python -m pip install -r requirements-lock.txt

The protocol-v2 rows record PyTorch `2.14.1+cpu` and commit `b5fbc1e` in their
notes. The CSV does not contain a full environment snapshot for those historical
runs; the lockfile documents a supported resolved environment for future runs, not
proof that every locked package version matches the original run environment.

## 2. Data
`data/processed/` is not in Git. Either copy Person 1's `processed.zip` contents there, or rebuild it from the raw
CropDP-KG files in `data/raw/`:

    python src/data/prepare_cropdp.py --raw data/raw --out data/processed --seed 42 --drop-relations EnglishName

(Person 1 should confirm that this reproduces the shared files byte for byte.) Check with `pytest tests/test_data.py`.

## 3. Run and verify

| Table row | Config | Needs |
|---|---|---|
| Popularity (sanity check) | `configs/popularity.yaml` | |
| RotatE alone (baseline 1) | `configs/rotate.yaml` | |
| R-GCN + DistMult (baseline 2) | `configs/rgcn_distmult.yaml` | |
| DistMult, no graph (ablation) | `configs/distmult_nograph.yaml` | |
| VRGAE, shared recipe | `configs/vrgae.yaml` | torch_geometric |
| VRGAE as trained by Person 3 | `configs/vrgae_checkpoint.yaml` | `results/vrgae_checkpoint.pt` (not in Git) |

    python src/eval/run_experiment.py --config configs/rotate.yaml      # trains, evaluates on TEST, appends rows
    python src/eval/make_table.py                                        # prefers protocol-v2 rows

For a new row with complete provenance, first verify its recorded inputs and then
rerun that specific run:

    python reproduce.py --run-id <run_id> --verify-only
    python reproduce.py --run-id <run_id> --tolerance 0.005

The runner's `--check` option and `src/eval/reproduce.py` compare metrics only. They
do not validate source/data/checkpoint provenance. They write no results or
checkpoints in check mode.

## 4. The shared protocol (the same for every model in the table)
- Evaluator: `src/eval/evaluator.py` only. Filtered tail prediction against train + val + test, ties get the average
  rank, AUC against the fixed `val_neg` / `test_neg`. The older private evaluation functions in `train.py` and
  `train_encoder.py` now call it.
- Negatives for training: `UniformNegativeSampler` (`src/data/prepare_cropdp.py`), known true triples filtered out,
  seeded.
- Model selection: best epoch on the FULL validation split; numbers in the table are TEST numbers, computed once.
- Seed: 42 for every row (a single seed; treat MRR differences below about 0.01 as ties).
- Deliberate difference: models run through the runner are trained in mini-batches with the batch's own edges hidden from
  message passing. `train.py` / `train_models.py` train with one full-batch step per epoch and the training edges visible;
  their VRGAE row is therefore reported separately (checkpoint row), not mixed with the shared-recipe row.

## 5. Recorded results and limitations
- `make_table.py` prefers the latest protocol-v2 run per config; v1 rows remain in
  `results.csv` for history and are marked `legacy-unverified`.
- The six protocol-v2 run groups retain the protocol, source commit, PyTorch build,
  selected epoch, and reported metrics. They are marked `recorded-unverified`
  because the processed dataset, complete environment/config hashes, training logs,
  and most checkpoints are not present.
- The Person 3 checkpoint row records a short checksum in its notes, but the
  checkpoint itself is absent. The short value is not a substitute for retaining
  and verifying the full checkpoint checksum.
- Consequently, the existing v2 rows can be inspected and used to regenerate the
  table, but cannot be independently reproduced from this checkout. The
  provenance-aware `reproduce.py` intentionally rejects them.
