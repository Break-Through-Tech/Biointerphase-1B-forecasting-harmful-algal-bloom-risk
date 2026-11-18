# Regional HAB forecasting with TabFM and LightGBM

The equal-weight blend is an experimental model. It reduced false alarms relative to tuned LightGBM in the 2020–2024 backtest, from 23 to 14, and caught one fewer onset. When its threshold targeted CatBoost's historical alert budget, the blend caught one more onset than CatBoost and generated 11 more false alarms.

## 1. Forecast target and inputs

The model forecasts next week's regional *Karenia brevis* severity. Class 0 is below 5,000 cells/L. Class 1 is 5,000 to 99,999 cells/L. Class 2 is at least 100,000 cells/L. Both models use the same 12 core features and labeled regional weeks. [Regional inputs and TabFM configuration](TabFM_Model_Development_Summary.md)

| Component | Configuration |
|---|---|
| TabFM | The 32-estimator classifier and labeled context from notebook 08 |
| Tuned LightGBM | 100 trees, learning rate 0.03, maximum depth 5, 15 leaves, balanced class weights, seed 42 |
| Blend | Average the class probabilities with fixed weights of 0.50 per model |

Notebook 09 checks that both models cover the same target weeks and orders the probabilities by class before averaging. The 50/50 weights were fixed for the experiment. LightGBM uses class weighting during fitting, and neither component is recalibrated before averaging. The blended probabilities serve as forecast scores for this comparison.

## 2. Evaluation and decisions

Notebook 09 reads TabFM's forecasts from notebook 08 and fits LightGBM on the same training folds. The historical comparison covers 156 labeled weeks from 2020 to 2024, including 13 onsets. The extended 2017–2024 backtest covers 313 weeks and 25 onsets. Each historical fold applies a four-week gap before its first labeled target week. The final 2025 test uses models fitted through 2024. [Ensemble notebook](notebooks/09_TabFM_LightGBM_Ensemble.ipynb)

The severity forecast chooses the class with the highest averaged probability. The alert policy adds the blended class 1 and 2 probabilities and selects the more probable of those classes if that sum reaches the cutoff. Below the cutoff it reports class 0. For each evaluation year from 2020 onward, the cutoff targets CatBoost's false-alarm count on earlier class-0 weeks. For 2025, that procedure produced a cutoff of 0.2868 using forecasts from 2017–2024. [TabFM evaluation and thresholds](TabFM_Model_Development_Summary.md#3-evaluation-design)

An onset is an elevated or severe week after the last observed class was 0. A severe miss is class 2 called class 0. A false alarm is class 0 called class 1 or 2.

## 3. Model comparison

| Model or alert policy, 2020–2024 | Onsets caught | Severe misses | False alarms | Cost 2:1 | Cost 3:1 | Macro-F1 | Onset AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0/13 | 2 | 12 | 16 | 18 | 0.677 | — |
| Tuned LightGBM | 2/13 | 2 | 23 | 27 | 29 | 0.586 | 0.141 |
| Tuned LightGBM, threshold set on earlier years | 6/13 | 0 | 53 | 53 | 53 | 0.518 | 0.141 |
| CatBoost warning at 0.35 | 6/13 | 0 | 33 | 33 | 33 | 0.636 | 0.205 |
| TabFM, highest-probability class | 0/13 | 4 | 9 | 17 | 21 | 0.627 | 0.154 |
| TabFM, threshold set on earlier years | 6/13 | 0 | 33 | 33 | 33 | 0.617 | 0.154 |
| TabFM + LightGBM, highest-probability class | 1/13 | 2 | 14 | 18 | 20 | 0.613 | 0.149 |
| TabFM + LightGBM, threshold set on earlier years | 7/13 | 0 | 44 | 44 | 44 | 0.565 | 0.149 |

The default blend landed between its two components on both onsets caught and false alarms. The thresholded blend caught the most onsets of the policies shown, with a higher cost than TabFM or CatBoost at both severe-miss penalties.

### Historical alert results by year

| Target year | Blend cutoff | Blend onsets caught | Blend false alarms | CatBoost onsets caught | CatBoost false alarms |
|---|---:|---:|---:|---:|---:|
| 2020 | 0.2082 | 2/6 | 18 | 1/6 | 13 |
| 2021 | 0.2624 | 1/2 | 2 | 2/2 | 2 |
| 2022 | 0.2624 | 1/1 | 1 | 1/1 | 1 |
| 2023 | 0.2624 | 1/2 | 4 | 0/2 | 7 |
| 2024 | 0.2402 | 2/2 | 19 | 2/2 | 10 |

Both alert policies caught both 2024 onsets. The blend issued nine more false alarms that year, accounting for most of the eleven additional false alarms in the pooled results. These 2024 figures come from the historical fold with a four-week gap. [Evaluation design](TabFM_Model_Development_Summary.md#3-evaluation-design)

### Extended backtest and 2025 test

Across 2017–2024, the highest-probability blend caught 3 of 25 onsets and produced 27 false alarms. LightGBM caught 6 and produced 38. The blend's onset AP was 0.259, compared with 0.243 for TabFM and 0.230 for LightGBM. The model-specific alert cutoffs start in 2020, so the comparison across all eight years uses the default decisions for those models.

| Model or policy, 2025 | Accuracy | Balanced accuracy | Macro-F1 | Severe misses | False alarms | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.962 | 0.659 | 0.637 | 0 | 1 | — |
| Tuned LightGBM | 0.904 | 0.589 | 0.600 | 0 | 3 | 0.300 |
| CatBoost warning at 0.35 | 0.904 | 0.637 | 0.626 | 0 | 4 | 0.309 |
| TabFM, highest-probability class | 0.962 | 0.659 | 0.637 | 0 | 1 | 0.207 |
| Blend, highest-probability class | 0.923 | 0.596 | 0.604 | 0 | 2 | 0.246 |
| Blend, threshold set on earlier years | 0.885 | 0.581 | 0.596 | 0 | 4 | 0.246 |

The default blend correctly classified 43 of 45 low weeks and 5 of 6 severe weeks. It labeled the single elevated week as severe and one severe week as elevated, so its severe-miss count was zero under the class-2-to-class-0 definition. The 52-week test contained zero onsets.

### Interpretation and limitations

The blend remains experimental. Its alert policy added one historical onset catch relative to the standalone TabFM and CatBoost alerts, at the cost of 11 more false alarms. The trial covers one fixed set of weights and uses a CatBoost reference cutoff selected on historical data. Further observed onsets would allow the team to compare those alert policies on new seasons.

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
