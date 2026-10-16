# Final modelling tables

These are the outputs of notebook 05 (`notebooks/05_feature_engineering.ipynb`), copied from `data/processed/` so you can train models without rerunning notebooks 02, 03 and 05. If those notebooks change, rerun them and copy the new outputs here.

| file | rows x cols | one row per |
|:-----|:------------|:------------|
| `model_table_weekly_fe_full.parquet` | 595 x 117 | week (regional table, all engineered features) |
| `model_table_weekly_fe_reduced.parquet` | 595 x 104 | week (regional table, redundant features dropped) |
| `site_week_table_fe.parquet` | 21,718 x 46 | sampling site and week |
| `feature_sets.json` | - | feature lists, label names and the split column |

Weeks run from 2015-01-05 to 2026-05-25. Features describe week t and the labels describe week t+1.

## Labels

- `y_class` (primary): 0 is below 5,000 cells/L, 1 is 5,000 to 99,999 and 2 is at least 100,000.
- `y_binary`: 1 when `y_class` is 1 or 2.
- `y_class_10k`, `y_class_box` and `y_class_mote_25km` are alternative definitions.

A week with no sample in the following week has no label (NaN). Drop those rows before training. In the regional table 486 of the 595 weeks have a label.

## Split

Use the `split` column, which is set by the forecast week:

| split | weeks |
|:------|:------|
| `train` | 2015 to 2023 |
| `validate` | 2024 |
| `test` | 2025 |
| `holdout` | 2026 (to May) |

Don't shuffle rows across splits.

## Choosing features

Start with `regional_core` (or `site_core` for the site table) in `feature_sets.json`, then add groups from `regional_candidates` / `site_candidates`. Never use the columns under `do_not_use` as features. They include the labels, the keys, the current-week class and sources with too little coverage.

Column definitions are in [`../docs/data_dictionary.md`](../docs/data_dictionary.md).
