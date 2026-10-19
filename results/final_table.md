## Table 1: overall and macro-average (test split)

| Model | MRR | Hits@1 | Hits@3 | Hits@10 | AUC | Macro MRR | Macro Hits@10 |
|---|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.1355 | 0.0747 | 0.1344 | 0.2533 | 0.7194 | 0.2734 | 0.4717 |
| RotatE alone (baseline 1) | 0.1963 | 0.1246 | 0.2235 | 0.3414 | 0.8418 | 0.3009 | 0.5247 |
| R-GCN + DistMult (baseline 2) | 0.0864 | 0.0268 | 0.0886 | 0.2039 | 0.8501 | 0.1518 | 0.3873 |
| DistMult, no graph (ablation) | 0.1581 | 0.0922 | 0.1818 | 0.2940 | 0.8720 | 0.2261 | 0.4237 |
| VRGAE-lite (ours, shared training recipe) | 0.0834 | 0.0433 | 0.0793 | 0.1627 | 0.9057 | 0.1534 | 0.2914 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0045 | 0.0005 | 0.0026 | 0.0093 | 0.4782 | 0.0037 | 0.0080 |

## Table 2: MRR per relation (test split)

| Model | RegionIs | ConditionIs | Damage | InflictHarmPart | ManifestAs | TemperIs* |
|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.2831 | 0.1924 | 0.1458 | 0.5635 | 0.0430 | 0.4127 |
| RotatE alone (baseline 1) | 0.4447 | 0.2519 | 0.3036 | 0.5473 | 0.0656 | 0.1925 |
| R-GCN + DistMult (baseline 2) | 0.2127 | 0.1252 | 0.0962 | 0.2546 | 0.0348 | 0.1872 |
| DistMult, no graph (ablation) | 0.4138 | 0.1603 | 0.2818 | 0.4293 | 0.0415 | 0.0297 |
| VRGAE-lite (ours, shared training recipe) | 0.2662 | 0.1301 | 0.0750 | 0.4317 | 0.0076 | 0.0101 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0045 | 0.0057 | 0.0073 | 0.0008 | 0.0037 | 0.0003 |
| n (test facts) | 156 | 203 | 387 | 132 | 1055 | 9 |

## Table 3: Hits@10 per relation (test split)

| Model | RegionIs | ConditionIs | Damage | InflictHarmPart | ManifestAs | TemperIs* |
|---|---|---|---|---|---|---|
| Popularity (sanity check) | 0.7115 | 0.4384 | 0.2894 | 0.7652 | 0.0701 | 0.5556 |
| RotatE alone (baseline 1) | 0.9167 | 0.4631 | 0.5556 | 0.7803 | 0.0995 | 0.3333 |
| R-GCN + DistMult (baseline 2) | 0.6538 | 0.2857 | 0.2300 | 0.5303 | 0.0682 | 0.5556 |
| DistMult, no graph (ablation) | 0.8141 | 0.3547 | 0.5065 | 0.6742 | 0.0815 | 0.1111 |
| VRGAE-lite (ours, shared training recipe) | 0.6090 | 0.2808 | 0.1576 | 0.6894 | 0.0114 | 0.0000 |
| VRGAE-lite (ours, as trained by Person 3) | 0.0064 | 0.0197 | 0.0155 | 0.0000 | 0.0066 | 0.0000 |
| n (test facts) | 156 | 203 | 387 | 132 | 1055 | 9 |

Seed(s): 42 (single seed: treat MRR differences below about 0.01 as ties). Best epoch chosen on the validation split; all numbers are filtered tail prediction on test.

\* TemperIs has only 9 test facts and is not reliable.

AUC is secondary: many fixed negatives have the wrong entity type, so AUC rewards type-matching more than fine-grained ranking. Rely on MRR and Hits@K.
