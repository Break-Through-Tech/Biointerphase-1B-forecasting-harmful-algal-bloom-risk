# Regional HAB forecasting with TabFM and LightGBM

The equal-weight blend is an experimental model. It reduced false alarms relative to tuned LightGBM in the 2020–2024 backtest, from 23 to 14, and caught one fewer onset. When its threshold targeted CatBoost's historical alert budget, the blend caught one more onset than CatBoost and generated 11 more false alarms.

## 1. Forecast target and inputs

The model forecasts next week's regional *Karenia brevis* severity. Class 0 is below 5,000 cells/L. Class 1 is 5,000 to 99,999 cells/L. Class 2 is at least 100,000 cells/L. Both models use the same 12 core features and labeled regional weeks. [Regional inputs and TabFM configuration](TabFM_Model_Development_Summary.md)

| Component | Configuration |
|---|---|
| TabFM | The 32-estimator classifier and labeled context from notebook 08 |
| Tuned LightGBM | 100 trees, learning rate 0.03, maximum depth 5, 15 leaves, balanced class weights, seed 42 |
| Blend | Average the class probabilities with fixed weights of 0.50 per model |

The notebook aligns probabilities by target week and class before blending. The mixture uses LightGBM's class-weighted scores without an additional calibration step. The fixed weights are a starting point for comparison.

## 2. Evaluation and decisions

Notebook 09 reads TabFM's forecasts from notebook 08 and fits LightGBM on the same training folds. The historical comparison covers 156 labeled weeks from 2020 to 2024, including 13 onsets. The extended 2017–2024 backtest covers 313 weeks and 25 onsets. Each historical fold applies a four-week gap before its first labeled target week. The final 2025 test uses models fitted through 2024. [Ensemble notebook](notebooks/09_TabFM_LightGBM_Ensemble.ipynb)

The severity forecast chooses the class with the highest averaged probability. The alert policy uses the sum of the blended class 1 and 2 probabilities. For each year, the cutoff targets CatBoost's false-alarm count on earlier class-0 weeks. Both decision rules use the same blended probabilities.

An onset is an elevated or severe week after the last observed class was 0. A severe miss is class 2 called class 0. A false alarm is class 0 called class 1 or 2.

## 3. Model comparison

| Model or alert policy, 2020–2024 | Onsets caught | Severe misses | False alarms | Macro-F1 |
|---|---:|---:|---:|---:|
| Persistence | 0/13 | 2 | 12 | 0.677 |
| Tuned LightGBM | 2/13 | 2 | 23 | 0.586 |
| CatBoost warning at 0.35 | 6/13 | 0 | 33 | 0.636 |
| TabFM, highest-probability class | 0/13 | 4 | 9 | 0.627 |
| TabFM, threshold set on earlier years | 6/13 | 0 | 33 | 0.617 |
| TabFM + LightGBM, highest-probability class | 1/13 | 2 | 14 | 0.613 |
| TabFM + LightGBM, threshold set on earlier years | 7/13 | 0 | 44 | 0.565 |

Across 2017–2024, the highest-probability blend caught 3 of 25 onsets and produced 27 false alarms. LightGBM caught 6 and produced 38. On the 52-week 2025 test, the highest-probability blend had zero severe misses, two false alarms, and macro-F1 of 0.604. LightGBM had zero severe misses, three false alarms, and macro-F1 of 0.600. The 2025 test had zero onsets.

Keep the blend in the comparison. The team selected CatBoost's 0.35 cutoff using historical data, so the backtest compares the alert policies retrospectively. The 13 historical onsets and the lack of onsets in 2025 limit how confidently the team can choose an alert model.

## 4. Reproduce and verify the saved forecast

Install the model dependencies and run notebook 08. Then run notebook 09 in the same checkout. It checks the input hashes and confirms the TabFM probabilities cover every evaluation week. [TabFM setup](TabFM_Model_Development_Summary.md#5-reproduce-and-verify-the-saved-forecast) · [Notebook 09](notebooks/09_TabFM_LightGBM_Ensemble.ipynb)

Notebook 09 writes the following files to `models/tabfm_lightgbm/`. Git ignores the directory.

| File | Contents |
|---|---|
| `config.json` | Model weights, input hashes, feature order, alert cutoffs, and the path to the TabFM context |
| `lightgbm_through_2024.txt` | The fitted LightGBM model in LightGBM's native format |
| `ensemble_probabilities.parquet` | The blended class probabilities for all 365 evaluation weeks |
| `metrics.csv` and `weekly_decisions.csv` | Summary metrics and per-week predictions |

The last cell reloads the native LightGBM model and reconstructs TabFM from notebook 08's saved context. It checks that the resulting blend reproduces the first 2025 forecast.
