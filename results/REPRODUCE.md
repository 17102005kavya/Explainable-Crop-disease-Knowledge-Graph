# Reproducing Part D (evaluator, baselines, VRGAE rows)

Everything in `results/final_table.md` comes from `results/results.csv`, and every row of the table comes from one
config in `configs/`. Rows written by the older team scripts (relation `all`, no `overall` row) are legacy rows and are
ignored by `make_table.py`.

## 1. Environment
1. Install PyTorch for your machine, then `pip install -r requirements.txt` (needs `torch_geometric`, `scikit-learn`, `pyyaml`).
2. The exact versions of the last full run are stored in `requirements-lock.txt`; each logged row also records `torch=...`
   and `commit=...` in its `notes` column.

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

    python src/eval/run_experiment.py --config configs/rotate.yaml     # trains, evaluates on TEST, appends rows
    python src/eval/make_table.py                                       # rebuilds results/final_table.md
    python src/eval/reproduce.py                                        # re-runs every config and compares with the log

`reproduce.py` writes nothing; it reports PASS when every metric of every relation is within 0.005 of the logged run
(same machine: expect differences of about 0.0000; other machines or torch versions can differ slightly).

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

## 5. What cannot be reproduced from Git alone
- `results/vrgae_checkpoint.pt` is gitignored. The row records the first 12 characters of its SHA-256 in `notes`, so a
  copy can be matched to the row. `python src/model/train.py` regenerates a checkpoint, but not necessarily the same bytes.
- Rows in `results.csv` written before protocol v2 (no `protocol=` in `notes`) used an inline sampler; they stay in the
  file but are superseded: `make_table.py` uses the newest run of each config.
