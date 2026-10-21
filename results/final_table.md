## Table 1: overall and macro-average (test split)

| Model | MRR | Hits@1 | Hits@3 | Hits@10 | AUC | Macro MRR | Macro Hits@10 |
|---|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.1355 | 0.0747 | 0.1344 | 0.2533 | 0.7194 | 0.2734 | 0.4717 |
| RotatE alone (baseline 1) | 0.2134 | 0.1509 | 0.2374 | 0.3393 | 0.8553 | 0.3502 | 0.5407 |
| R-GCN + DistMult (baseline 2) | 0.1158 | 0.0623 | 0.1112 | 0.2317 | 0.8789 | 0.2205 | 0.5014 |
| DistMult, no graph (ablation) | 0.1635 | 0.1004 | 0.1818 | 0.2930 | 0.8735 | 0.2369 | 0.4283 |
| VRGAE-lite (ours, shared training recipe) | 0.0793 | 0.0396 | 0.0721 | 0.1643 | 0.9062 | 0.1509 | 0.3143 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0045 | 0.0005 | 0.0026 | 0.0093 | 0.4782 | 0.0037 | 0.0080 |

## Table 2: MRR per relation (test split)

| Model | RegionIs | ConditionIs | Damage | InflictHarmPart | ManifestAs | TemperIs* |
|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.2831 | 0.1924 | 0.1458 | 0.5635 | 0.0430 | 0.4127 |
| RotatE alone (baseline 1) | 0.5269 | 0.2502 | 0.3339 | 0.5634 | 0.0707 | 0.3561 |
| R-GCN + DistMult (baseline 2) | 0.2196 | 0.1658 | 0.1011 | 0.5407 | 0.0418 | 0.2543 |
| DistMult, no graph (ablation) | 0.4240 | 0.1644 | 0.2889 | 0.4740 | 0.0410 | 0.0293 |
| VRGAE-lite (ours, shared training recipe) | 0.2548 | 0.1044 | 0.0775 | 0.4224 | 0.0066 | 0.0399 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0045 | 0.0057 | 0.0073 | 0.0008 | 0.0037 | 0.0003 |
| n (test facts) | 156 | 203 | 387 | 132 | 1055 | 9 |

## Table 3: Hits@10 per relation (test split)

| Model | RegionIs | ConditionIs | Damage | InflictHarmPart | ManifestAs | TemperIs* |
|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.7115 | 0.4384 | 0.2894 | 0.7652 | 0.0701 | 0.5556 |
| RotatE alone (baseline 1) | 0.9231 | 0.4286 | 0.5530 | 0.7955 | 0.0995 | 0.4444 |
| R-GCN + DistMult (baseline 2) | 0.6346 | 0.4187 | 0.2196 | 0.7803 | 0.0664 | 0.8889 |
| DistMult, no graph (ablation) | 0.8269 | 0.3448 | 0.5065 | 0.7045 | 0.0758 | 0.1111 |
| VRGAE-lite (ours, shared training recipe) | 0.6603 | 0.2512 | 0.1550 | 0.6970 | 0.0114 | 0.1111 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0064 | 0.0197 | 0.0155 | 0.0000 | 0.0066 | 0.0000 |
| n (test facts) | 156 | 203 | 387 | 132 | 1055 | 9 |

Protocol: v2 (shared filtered negative sampler and evaluator). Seed(s): 42 (single seed: treat MRR differences below about 0.01 as ties). Best epoch chosen on the full validation split; all numbers are filtered tail prediction on test. Earlier protocol-v1 rows are retained in results.csv for history but excluded from this comparison.

\* TemperIs has only 9 test facts and is not reliable.

AUC is secondary: many fixed negatives have the wrong entity type, so AUC rewards type-matching more than fine-grained ranking. Rely on MRR and Hits@K.
