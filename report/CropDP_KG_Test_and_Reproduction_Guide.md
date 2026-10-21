# CropDP-KG Test and Reproduction Guide

**Status:** commands below describe how to reproduce/check the project. They are not evidence that the commands passed. At the 2026-10-20 review, raw/processed data, checkpoints, logs, and `pytest` were unavailable in the checkout/environment.

## 1. Environment

The direct dependencies are pinned in `requirements.txt`; `requirements-lock.txt` is the
resolved transitive lock for Python 3.14.6 on Windows x64. Create a fresh environment
with that interpreter, then install the lock:

```powershell
python -m pip install -r requirements-lock.txt
```

The lock includes the standard package-index PyTorch build, not a promise of CPU-only
execution. The run log records the installed PyTorch build and CUDA runtime; use the
official PyTorch CPU wheel index if CPU-only execution is required, and retain the
resulting environment details with the run. The lock is tied to the Python/platform
target stated above. To update it after changing direct dependencies:

```powershell
python -m pip install pip-tools
python -m piptools compile --allow-unsafe --no-annotate --no-emit-index-url --output-file requirements-lock.txt requirements.txt
```

## 2. Data preparation

Acquire the authorized raw CropDP-KG `relation*.csv` files described in `data\README.md` and place them in `data\raw`.

Inspect the files first:

```powershell
python src\data\prepare_cropdp.py --raw data\raw --inspect
```

Generate processed data using the documented intended six-relation setup:

```powershell
python src\data\prepare_cropdp.py --raw data\raw --out data\processed --seed 42 --drop-relations EnglishName
```

Review `data\processed\meta.json`, `data_report.md`, `entities.csv`, `relations.csv`, `provenance.csv`, and `conflicts.csv`. README values (11,417 entities; six relations; 19,422 triples; train/validation/test sizes 15,538/1,942/1,942) are expectations stated in documentation, not guaranteed results. Treat generated files as authoritative and compare before using those figures.

The synthetic demo can exercise the preparation code without the real dataset:

```powershell
python src\data\prepare_cropdp.py --demo data\demo_raw
python src\data\prepare_cropdp.py --raw data\demo_raw --out data\demo_processed --seed 42
```

Demo data does not verify CropDP-KG counts or model performance.

## 3. Tests

From the repository root, after dependencies are installed:

```powershell
python -m pytest -p no:cacheprovider -q
```

Focused suites:

```powershell
python -m pytest -p no:cacheprovider -q tests\test_data.py
python -m pytest -p no:cacheprovider -q tests\test_encoder.py
python -m pytest -p no:cacheprovider -q tests\test_decoder.py
python -m pytest -p no:cacheprovider -q tests\test_eval.py
python -m pytest -p no:cacheprovider -q tests\test_vrgae_adapter.py
```

No pass/fail count is promised here. Several tests skip when `data\processed` or a checkpoint is missing; the VRGAE adapter test also requires the relevant ML dependencies. Save actual test output with the environment details.

## 4. Training and evaluation commands

These scripts can create/overwrite files at the paths supplied to them. Before running, choose new output paths or preserve existing artifacts.

### Person 2 encoder smoke/full training

```powershell
python src\encoder\train_encoder.py --config configs\default.yaml --epochs 50 --eval-every 10 --checkpoint results\encoder_checkpoint.pt --log results\encoder_train.log
```

The encoder training path uses a custom validation ranker and defaults to evaluating at most 500 validation triples. Use `--eval-limit 0` for a full-validation checkpoint evaluation:

```powershell
python src\encoder\train_encoder.py --eval-only --checkpoint results\encoder_checkpoint.pt --eval-limit 0
```

### Baselines

```powershell
python src\eval\run_experiment.py --config configs\popularity.yaml
python src\eval\run_experiment.py --config configs\rotate.yaml
python src\eval\run_experiment.py --config configs\rgcn_distmult.yaml
python src\eval\run_experiment.py --config configs\distmult_nograph.yaml
```

Each run appends rows to `results\results.csv` and saves a checkpoint under
`results\checkpoints`. The CSV rows record a protocol ID, source commit and code hash,
dirty-source flag, config hash and effective config, processed-data hash, Python and
package versions, selected epoch, checkpoint path/hash, and (for checkpoint evaluation)
the input checkpoint path/hash.

### VRGAE

The repository provides multiple VRGAE paths with different training semantics. Do not treat their output as a like-for-like comparison without first choosing and documenting one protocol.

```powershell
python src\eval\run_experiment.py --config configs\vrgae.yaml
```

The legacy full-batch trainer is separate:

```powershell
python src\model\train.py --config configs\default.yaml --epochs 200 --eval-every 10 --checkpoint results\vrgae_checkpoint.pt --log results\vrgae_train.log
```

`src\model\train_models.py` also contains another path and may write checkpoints/results. Inspect its configured output names before use.

### Table generation

```powershell
python src\eval\make_table.py
```

The table generator consumes recorded CSV rows; it does not retrain or validate models.

### Reproduce or verify a logged run

First ensure the processed data, original input checkpoint (if applicable), recorded
source contents, and locked environment are available. Validate provenance without training:

```powershell
python reproduce.py --run-id <run_id> --verify-only
```

Rerun the experiment and compare all overall, macro, and per-relation metrics to the
recorded rows. The default absolute tolerance is 0.005:

```powershell
python reproduce.py --run-id <run_id> --tolerance 0.005
```

The reproduction command writes only to a temporary directory. It fails rather than
guessing when recorded provenance, data, source checkpoint, or environment differs.
Rows from earlier runs are marked `legacy-unverified`: their original data, source
identity, environment, and checkpoints were not retained, so they cannot be made
reproducible retroactively.

## 5. Test coverage by area

- `tests\test_data.py`: processed array shapes/ranges, split overlap, entity coverage, graph construction, and fixed-negative checks.
- `tests\test_encoder.py`: synthetic encoder shape/gradient/scorer checks and data/checkpoint-dependent checks.
- `tests\test_decoder.py`: variational layer, KL, RotatE scoring/inverse/gradients, losses, and integrated-model checks.
- `tests\test_eval.py`: filtering, tie handling, metric keys, and AUC behavior.
- `tests\test_vrgae_adapter.py`: short synthetic shared-recipe training/evaluation test.
- `tests\test_provenance.py`: data hashing, protocol labels, legacy-run rejection, and metric tolerance checks.

Test definitions are not equivalent to a passing test run. Record actual output from the environment where these commands are executed.

## 6. Result reproduction requirements

Before describing a metric as reproduced, retain:

1. Dataset source and exact processed data identity, including entity/relation mapping tables.
2. Git commit or source revision, full config, random seed, and environment versions.
3. Training log, selected validation epoch, checkpoint, and checkpoint checksum.
4. Test evaluation output, evaluator version, candidate set, filtering protocol, and fixed-negative identity for AUC.
5. The generated CSV rows and final table.

The historical rows currently in `results/results.csv` predate provenance logging and
are explicitly marked `legacy-unverified`. The data arrays and historical checkpoints
are also absent from this checkout. New runs record the required provenance and
checkpoint artifacts automatically; this does not retroactively validate historical
metrics or establish that the PDF's protocol-v2 values match the checked-in result rows.
