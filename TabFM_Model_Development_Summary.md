# Regional HAB forecasting with TabFM

TabFM belongs in the regional comparison as a standalone model. On the 2020–2024 backtest, its highest-probability class forecasts produced 9 false alarms and missed all 13 onsets. With a risk threshold set on earlier years, it caught 6 of 13 onsets with 33 false alarms and zero severe misses. This matches the counts for the team's CatBoost warning, though each model caught two onsets the other missed.

## 1. Forecast target and data

The model predicts next week's *Karenia brevis* severity for the region from Anna Maria Island to Boca Grande.

| Class | Severity | Cell count |
|---|---|---:|
| 0 | Low | Below 5,000 cells/L |
| 1 | Elevated | 5,000–99,999 cells/L |
| 2 | Severe | At least 100,000 cells/L |

The regional model table has 595 weeks, of which 486 have a next-week label. An unsampled target week has an unknown label and stays out of fitting and evaluation. Notebook 08 uses the same 12 core features as the team's boosting comparison. [Regional modeling table](data/final/README.md) · [Boosting model summary](Regional_HAB_Boosting_Model_Development_Summary.md)

| Feature group | Columns |
|---|---|
| Last observed bloom state | `last_class`, `last_log_cells`, `weeks_since_obs` |
| Recent change and duration | `last_class_prev`, `last_log_cells_chg1`, `class_max_4wk`, `weeks_since_bloom`, `pos_weeks_12wk`, `class_run_weeks` |
| Sampling and season | `n_habsos_samples`, `season_sin`, `season_cos` |

## 2. TabFM configuration

Google's pretrained classifier reads the labeled training rows as context and predicts class probabilities for new rows. The pretrained weights stay fixed. This gives the team a comparison with a model that can use a small historical table without fitting new weights to it. [Google Research's TabFM overview](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/)

| Setting | Value |
|---|---|
| Checkpoint | `google/tabfm-1.0.0-pytorch`, classification |
| Package | `tabfm[pytorch]==1.0.1` |
| Configuration | Standard classifier with 32 estimators and all training rows as context |
| Missing values | Add indicators for features with missing values in the training fold |
| Seed and precision | 42, float32 |

## 3. Evaluation design

- **Historical comparison, 2020–2024.** Forecast 156 labeled weeks with 13 onsets using the splits from notebook 07. For each year, the training cutoff is four weeks before the first labeled target week.
- **Extended backtest, 2017–2024.** Apply the same split rule to 313 labeled weeks with 25 onsets. The 2017–2019 forecasts also supply the first alert threshold.
- **Final test, 2025.** Fit the context on 416 labeled weeks through 2024 and evaluate on 52 labeled weeks. Keep the 2026 holdout outside this experiment.

An onset is a target week at class 1 or 2 after the last observed class was 0. A severe miss is class 2 called class 0. A false alarm is class 0 called class 1 or 2. The cost scores add the false-alarm count to two or three times the severe-miss count. Notebook 08 also reports macro-F1, balanced accuracy, onset average precision, and log loss.

TabFM predicts the highest-probability class in the default comparison. The alert policy uses `P(class 1) + P(class 2)`. Each year from 2020 onward, it chooses a cutoff on earlier class-0 forecasts to target the false-alarm count of CatBoost at 0.35 on those same earlier weeks. The realized count can change when the cutoff is applied to a new year.

## 4. Model comparison

| Model, 2020–2024 | Onsets caught | Severe misses | False alarms | Macro-F1 |
|---|---:|---:|---:|---:|
| Persistence | 0/13 | 2 | 12 | 0.677 |
| Tuned LightGBM | 2/13 | 2 | 23 | 0.586 |
| CatBoost warning at 0.35 | 6/13 | 0 | 33 | 0.636 |
| TabFM, highest-probability class | 0/13 | 4 | 9 | 0.627 |
| TabFM, threshold set on earlier years | 6/13 | 0 | 33 | 0.617 |

In the extended backtest, the default TabFM forecast caught 2 of 25 onsets and produced 20 false alarms. Tuned LightGBM caught 6 and produced 38. On the 2025 test, default TabFM and persistence both had one false alarm, zero severe misses, and macro-F1 of 0.637. The 2025 test had zero onsets, so onset recall is undefined.

The alert results justify a comparison with CatBoost on future observed onsets. The 13 historical onsets give limited precision, and the published CatBoost cutoff was selected using historical data. The weekly maximum also depends on when and where sampling occurred. Keep tuned LightGBM as the current severity reference and CatBoost as the early warning reference.

## 5. Reproduce and verify the saved forecast

Use Python 3.11 in a project checkout with Git LFS installed. Run these commands from the repository root.

```sh
git lfs pull --include="data/final/feature_sets.json"
python -m pip install -r requirements-tabfm.txt
```

Open notebook 08 in Jupyter or VS Code with that Python environment and run all cells. TabFM uses CUDA when available and CPU otherwise. The first run needs access to fetch the public checkpoint. Subsequent runs can use the cached weights. [Notebook 08](notebooks/08_TabFM_Regional_HAB.ipynb)

The notebook writes the following files to `models/tabfm/`. Git ignores the directory.

| File | Contents |
|---|---|
| `config.json` | Input hashes, feature order, checkpoint, package versions, and alert cutoffs |
| `tabfm_context_through_2024.parquet` | The 416 labeled context rows used for the 2025 fit |
| `tabfm_probabilities.parquet` | The 365 weekly forecasts, indexed by target week, for notebook 09 |
| `metrics.csv` and `weekly_decisions.csv` | Summary metrics and per-week predictions |

The last cell reloads the saved context and rebuilds the classifier with the public checkpoint. It checks that the first 2025 forecast matches the backtest probabilities.
