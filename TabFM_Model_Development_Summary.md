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

| Dataset split by target year | Labeled weeks | Low | Elevated | Severe |
|---|---:|---:|---:|---:|
| 2015–2023 training | 364 | 185 | 59 | 120 |
| 2024 validation | 52 | 38 | 5 | 9 |
| 2025 test | 52 | 45 | 1 | 6 |
| 2026 holdout | 18 | 18 | 0 | 0 |

The phase split describes the dataset. The historical backtests below use smaller training sets where the four-week gap applies. The final 2025 fit uses both the training and validation rows.

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
| Missing values | Keep missing predictors and add indicators for columns with missing values in that fold's training data |
| Seed and precision | 42, float32 |

## 3. Evaluation design

- **Historical comparison, 2020–2024.** Forecast 156 labeled weeks with 13 onsets using the splits from notebook 07. For each year, the training cutoff is four weeks before the first labeled target week.
- **Extended backtest, 2017–2024.** Apply the same split rule to 313 labeled weeks with 25 onsets. The 2017–2019 forecasts also supply the first alert threshold.
- **Final test, 2025.** Fit the context on 416 labeled weeks through 2024 and evaluate on 52 labeled weeks. Keep the 2026 holdout outside this experiment.

The 2024 historical fold uses 363 earlier weeks after the four-week gap. The separate 2024 validation experiment in the boosting summary uses all 364 training weeks. Use the historical results from each summary when comparing models on the same fit and evaluation windows. [Boosting backtests](Regional_HAB_Boosting_Model_Development_Summary.md#16-historical-catboost-backtesting)

An onset is a target week at class 1 or 2 after the last observed class was 0. A severe miss is class 2 called class 0. A false alarm is class 0 called class 1 or 2. The cost scores add the false-alarm count to two or three times the severe-miss count. Notebook 08 also reports macro-F1, balanced accuracy, onset average precision, and log loss.

TabFM predicts the highest-probability class in the default comparison. The alert policy uses `P(class 1) + P(class 2)`. Each year from 2020 onward, it chooses a cutoff on earlier class-0 forecasts to target the false-alarm count of CatBoost at 0.35 on those same earlier weeks. The realized count can change when the cutoff is applied to a new year.

When the risk score reaches the cutoff, the policy reports the more probable of the two elevated classes. Below the cutoff it reports class 0. In 2025, the TabFM cutoff was 0.2383, calculated from forecasts issued for 2017–2024.

## 4. Model comparison

| Model, 2020–2024 | Onsets caught | Severe misses | False alarms | Cost 2:1 | Cost 3:1 | Macro-F1 | Onset AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0/13 | 2 | 12 | 16 | 18 | 0.677 | — |
| Tuned LightGBM | 2/13 | 2 | 23 | 27 | 29 | 0.586 | 0.141 |
| Tuned LightGBM, threshold set on earlier years | 6/13 | 0 | 53 | 53 | 53 | 0.518 | 0.141 |
| CatBoost warning at 0.35 | 6/13 | 0 | 33 | 33 | 33 | 0.636 | 0.205 |
| TabFM, highest-probability class | 0/13 | 4 | 9 | 17 | 21 | 0.627 | 0.154 |
| TabFM, threshold set on earlier years | 6/13 | 0 | 33 | 33 | 33 | 0.617 | 0.154 |

Onset average precision (AP) measures how the risk scores rank new onsets among weeks whose last observed class was 0. Changing the cutoff changes the class decisions while leaving the score ranking the same. TabFM and CatBoost each caught six onsets after thresholding, with four shared catches, but CatBoost ranked onsets higher on this window.

### Historical alert results by year

| Target year | Labeled weeks | TabFM cutoff | TabFM onsets caught | TabFM false alarms | CatBoost onsets caught | CatBoost false alarms |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 52 | 0.2310 | 2/6 | 20 | 1/6 | 13 |
| 2021 | 27 | 0.2842 | 1/2 | 2 | 2/2 | 2 |
| 2022 | 8 | 0.2842 | 1/1 | 1 | 1/1 | 1 |
| 2023 | 17 | 0.2842 | 0/2 | 4 | 0/2 | 7 |
| 2024 | 52 | 0.2530 | 2/2 | 6 | 2/2 | 10 |

The pooled false-alarm totals hide variation across years. TabFM generated seven more false alarms than CatBoost in 2020 and four fewer in 2024. In the extended 2017–2024 backtest, the default TabFM forecast caught 2 of 25 onsets and produced 20 false alarms. Tuned LightGBM caught 6 and produced 38.

### Final test in 2025

| Model or policy | Accuracy | Balanced accuracy | Macro-F1 | Severe misses | False alarms | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.962 | 0.659 | 0.637 | 0 | 1 | — |
| Tuned LightGBM | 0.904 | 0.589 | 0.600 | 0 | 3 | 0.300 |
| CatBoost warning at 0.35 | 0.904 | 0.637 | 0.626 | 0 | 4 | 0.309 |
| TabFM, highest-probability class | 0.962 | 0.659 | 0.637 | 0 | 1 | 0.207 |
| TabFM, threshold set on earlier years | 0.923 | 0.644 | 0.630 | 0 | 3 | 0.207 |

Default TabFM and persistence made identical class decisions. TabFM correctly classified 44 of 45 low weeks and all six severe weeks. It predicted class 2 for the single class-1 week. The 2025 test contained zero onsets, so onset recall and onset AP are undefined.

### Interpretation and limitations

The historical results support keeping TabFM as a standalone candidate alongside the team's LightGBM severity forecast and CatBoost alert policy. The comparison covers only 13 onsets in 2020–2024, and CatBoost's 0.35 cutoff was selected on that historical period. The 2025 test provides evidence about false alarms and severity errors during that year. Comparing early-warning ability requires more observed onsets.

The target is the regional weekly maximum observed cell count, which depends on sampling coverage. The notebooks evaluate this regional target using the agreed core features. Site-level performance and the contribution of additional environmental predictors are outside this experiment.

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
