# CropDP-KG data report

Seed 42; val 0.1, test 0.1; near-duplicate merge: False; dropped relations: ['EnglishName']

## Cleaning

- raw rows: 21961
- kept: 19422
- dropped_by_flag: 2134
- dropped_missing: 402
- dropped_duplicate: 3
- entities: 11417, relations: 6, triples: 19422

## Per relation

| relation | triples | eligible for val/test | train | val | test |
|---|---|---|---|---|---|
| RegionIs | 1562 | 1405 | 1250 | 156 | 156 |
| ConditionIs | 2031 | 1338 | 1625 | 203 | 203 |
| Damage | 3870 | 2941 | 3096 | 387 | 387 |
| InflictHarmPart | 1321 | 1116 | 1057 | 132 | 132 |
| ManifestAs | 10549 | 2554 | 8439 | 1055 | 1055 |
| TemperIs | 89 | 52 | 71 | 9 | 9 |

## Entity types (multi-hot counts)

- PestDisease: 2204
- Crop: 623
- Symptom: 7816
- Condition: 576
- Part: 130
- Area: 59
- Temperature: 23
- EnglishName: 0

## Graph shape

- mean degree 3.40, median 1, max 331
- entities with degree 1: 7657 (67.1%)
- connected components (train graph): 332; largest has 9979 entities (87.4%); components with <=3 entities: 145

## Conflicts (kept and flagged, review by hand)

- tail_in_multiple_types: 13
- near_duplicate_entities: 2
- head_and_tail_type_clash: 1
- pair_in_multiple_relations: 1

## Leakage checks (overlaps must be 0)

- train_val_overlap: 0
- train_test_overlap: 0
- val_test_overlap: 0
- val_entities_missing_in_train: 0
- test_entities_missing_in_train: 0
- val_pair_seen_in_train_other_relation: 0
- val_reversed_pair_in_train: 0
- test_pair_seen_in_train_other_relation: 0
- test_reversed_pair_in_train: 0
- moved_to_train_for_pair_leak: 0

## Warnings

- none
