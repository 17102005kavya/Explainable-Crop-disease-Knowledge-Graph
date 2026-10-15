# BTP - VRGAE-lite on CropDP-KG (Phase 1)

Link prediction on the CropDP-KG crop disease and pest knowledge graph.
Phase 1 (reduced): R-GCN encoder + variational latent + RotatE decoder, two baselines, one seed, small manual grid.

## Who owns what
| Person | Part | Folder | Branch |
|---|---|---|---|
| Person 1 | A. Data & KG prep | `src/data/` | `data` |
| Person 2 | B. Encoder (R-GCN) | `src/encoder/` | `encoder` |
| Person 3 | C. Latent, decoder, training | `src/model/` | `latent-decoder` |
| Person 4 | D. Evaluation & baselines | `src/eval/` | `eval` |

Review ring: Person 1 reviews Person 2, 2 reviews 3, 3 reviews 4, 4 reviews 1. Nobody commits straight to `main`.

## Data (from Person 1, in `data/processed/`)
- 11,417 entities, 6 relations, 19,422 triples (EnglishName relation dropped). Train 15,538 / val 1,942 / test 1,942.
- Relation ids 0-5: RegionIs, ConditionIs, Damage, InflictHarmPart, ManifestAs, TemperIs.
- Every triple is `(head, relation, tail)` as int64; heads are always pest/disease nodes.
- `graph_edge_index.npy` / `graph_edge_type.npy`: train-only message-passing graph, inverse edges already added
  (inverse of relation r has id r + 6, so 12 relation ids). Never add val/test edges to it.
- `node_types.npy`: [11417, 8] multi-hot entity types (the EnglishName column is all zeros).
- `val_neg.npy`, `test_neg.npy`: one fixed filtered negative per positive, for AUC.
- Read `data/processed/data_report.md` and `meta.json` first.

## Interface contracts (do not change without telling the team)
```
Encoder   (Person 2):  RGCNEncoder.forward(x, edge_index, edge_type) -> H            # [N, d], d = 128
Latent    (Person 3):  VariationalLatent.forward(H) -> (z, mu, logvar)               # each [N, latent_dim]
Decoder   (Person 3):  RotatEDecoder.score(z, h, r, t) -> scores                      # [B]
                       RotatEDecoder.score_all_tails(z, h, r) -> scores               # [B, N]
Scorer    (everyone):  score_all_tails(h: LongTensor[B], r: LongTensor[B]) -> FloatTensor[B, N]
Evaluator (Person 4):  evaluate_ranking(score_all_tails, triples, known_triples, num_entities) -> dict
```
Every model, baselines included, is evaluated through the same `evaluate_ranking`
(filtered MRR, Hits@1/3/10, tail prediction by default) and `evaluate_auc`. Report per relation and the average.

## Quick start
```
pip install -r requirements.txt
pytest tests/test_data.py
# only to regenerate the data:
python src/data/prepare_cropdp.py --raw data/raw --out data/processed --seed 42 --drop-relations EnglishName
```
One YAML per run in `configs/`; log every run in `results/results.csv`.
