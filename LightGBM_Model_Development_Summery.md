# LightGBM Boosting Model — Regional HAB Forecasting

## Overview

This notebook develops a **LightGBM model** to predict next-week regional harmful algal bloom (HAB) severity along the Sarasota focus coast.

The model uses information from week **t** to predict HAB severity during week **t+1**.

The final model predicts one of three classes:

- **Class 0:** < 5,000 cells/L
- **Class 1:** 5,000–99,999 cells/L
- **Class 2:** >= 100,000 cells/L

---

## 1. Dataset

Main dataset:

```text
data/final/model_table_weekly_fe_reduced.parquet
```

Dataset size:

- **595 weekly rows**
- **104 columns**
- **486 labeled rows**

### Time-Based Split

| Split | Target Years | Labeled Rows |
|---|---|---:|
| Train | 2015–2023 | 364 |
| Validation | 2024 | 52 |
| Test | 2025 | 52 |
| Holdout | 2026 | 18 |

A random split was not used because this is a forecasting problem.

---

## 2. Final 12 Core Features

The main LightGBM model uses these 12 regional features:

1. `last_class`
2. `last_log_cells`
3. `last_class_prev`
4. `last_log_cells_chg1`
5. `weeks_since_obs`
6. `class_max_4wk`
7. `n_habsos_samples`
8. `weeks_since_bloom`
9. `pos_weeks_12wk`
10. `class_run_weeks`
11. `season_sin`
12. `season_cos`

LightGBM can handle missing predictor values directly, so missing values were not automatically replaced with zero.

---

## 3. First LightGBM Model

The first LightGBM model used the 12 core features.

This gave a starting point for comparing later improvements.

### Initial Core Model — 2024 Validation

- Accuracy: **0.846**
- Balanced Accuracy: **0.728**
- Macro F1: **0.698**

---

## 4. Handling Class Imbalance

The training classes were not evenly distributed.

To give the smaller classes more importance, another LightGBM model was trained using:

```python
class_weight="balanced"
```

### Balanced Core Model — 2024 Validation

- Accuracy: **0.865**
- Balanced Accuracy: **0.737**
- Macro F1: **0.715**

The balanced model performed better, so balanced class weighting was kept.

---

## 5. Environmental Feature Experiments

Several environmental feature groups were added to the 12 core features to see if they improved prediction.

Groups tested:

- Temperature
- Chlorophyll
- Wind
- Bay nutrients
- Rain
- Combined major environmental features

### Results

| Feature Set | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Core 12 Balanced | 0.865 | 0.737 | 0.715 |
| Core + Temperature | 0.808 | 0.624 | 0.615 |
| Core + Chlorophyll | 0.769 | 0.665 | 0.638 |
| Core + Wind | 0.808 | 0.710 | 0.670 |
| Core + Nutrients | 0.827 | 0.661 | 0.648 |
| Core + Rain | 0.827 | 0.691 | 0.673 |
| Core + Major Environmental Features | 0.788 | 0.615 | 0.604 |

### Decision

None of the environmental feature groups improved validation performance over the core features.

The final LightGBM therefore kept the **12 core regional features**.

This does not mean environmental variables are not important to HAB formation. It only means they did not improve this regional next-week model in the current dataset.

---

## 6. Hyperparameter Tuning

The LightGBM hyperparameters were tuned using **time-aware cross-validation** on the training period.

The main tuning metric was **Macro F1**.

### Final Hyperparameters

```python
LGBMClassifier(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    max_depth=5,
    n_estimators=100,
    num_leaves=15,
    class_weight="balanced",
    random_state=42
)
```

### Tuned LightGBM — 2024 Validation

- Accuracy: **0.865**
- Balanced Accuracy: **0.737**
- Macro F1: **0.717**

This became the selected LightGBM model.

---

## 7. Persistence Baseline

A persistence baseline was used for comparison.

Persistence simply predicts:

> next week will have the same class as the most recently observed class.

### 2024 Validation

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Persistence | 0.846 | 0.613 | 0.622 |
| **Tuned LightGBM** | **0.865** | **0.737** | **0.717** |

The tuned LightGBM performed better across the three classes.

---

# 8. Advisor-Priority Metrics

After advisor feedback, the notebook was updated to evaluate the model using metrics that focus more directly on **early warning**.

## Onset Recall

A true onset happens when:

```text
previous class = 0
and
next week's true class = 1 or 2
```

Onset recall tells us:

> Of the true new-risk events, how many did the model warn about?

Higher is better.

---

## Bloom Miss

A bloom miss is:

```text
True Class 2 → Predicted Class 0
```

This is an important error because a severe bloom is predicted as low risk.

Lower is better.

---

## False Alarm

A false alarm is:

```text
True Class 0 → Predicted Class 1 or 2
```

This means the model warns about risk when the actual week is low risk.

Lower is better.

---

## Risk-Sensitive Cost

The advisor suggested that missing a bloom should cost more than a false alarm.

Two versions were added:

```text
2:1 Cost = 2 × Bloom Misses + False Alarms
```

```text
3:1 Cost = 3 × Bloom Misses + False Alarms
```

Lower is better.

These are evaluation scores, not probabilities.

---

## 9. 2024 Advisor-Priority Validation

The main comparison is between the tuned LightGBM and persistence on the 2024 validation set.

| Metric | Persistence | Tuned LightGBM |
|---|---:|---:|
| Accuracy | 0.846 | **0.865** |
| Balanced Accuracy | 0.613 | **0.737** |
| Macro F1 | 0.622 | **0.717** |
| True Onsets | 2 | 2 |
| Onsets Caught | 0 | **1** |
| Onset Recall | 0.00 | **0.50** |
| Bloom Misses | 0 | 0 |
| False Alarms | **1** | 3 |
| Cost 2:1 | **1** | 3 |
| Cost 3:1 | **1** | 3 |

### Interpretation

The tuned LightGBM:

- caught **1 of 2** new onset events,
- while persistence caught **0 of 2**,
- had **0 severe bloom misses**,
- but produced more false alarms.

This shows the main trade-off:

> LightGBM gave better early-warning ability, but it was less conservative than persistence.

Only **2 onset events** occurred in 2024, so the 50% onset recall should be interpreted carefully.

---

## 10. Final 2025 Test

After model selection, the final LightGBM was retrained using:

- training data,
- plus 2024 validation data.

It was then tested on 2025.

### 2025 Class Distribution

- Class 0: **45**
- Class 1: **1**
- Class 2: **6**

### Final LightGBM Results

| Metric | Score |
|---|---:|
| Accuracy | 0.904 |
| Balanced Accuracy | 0.589 |
| Weighted Precision | 0.962 |
| Weighted Recall | 0.904 |
| Weighted F1 | 0.932 |
| Macro Precision | 0.611 |
| Macro Recall | 0.589 |
| Macro F1 | 0.600 |

The model correctly identified:

- **42 of 45** Class 0 weeks
- **5 of 6** Class 2 weeks
- **0 of 1** Class 1 weeks

No true Class 2 week was predicted as Class 0.

---

## 11. 2025 Advisor-Priority Metrics

The 2025 test set contained **0 true onset events**.

Because there were no true onsets:

```text
Onset Recall = undefined / NaN
```

This is correct. It should not be changed to zero.

### LightGBM vs Persistence — 2025

| Metric | Persistence | Final LightGBM |
|---|---:|---:|
| Accuracy | **0.962** | 0.904 |
| Balanced Accuracy | **0.659** | 0.589 |
| Macro F1 | **0.637** | 0.600 |
| True Onsets | 0 | 0 |
| Onsets Caught | 0 | 0 |
| Onset Recall | NaN | NaN |
| Bloom Misses | 0 | 0 |
| False Alarms | **1** | 3 |
| Cost 2:1 | **1** | 3 |
| Cost 3:1 | **1** | 3 |

### Interpretation

Persistence performed better on the 2025 test year.

However, 2025 cannot tell us whether one model is better at early onset detection because there were **no onset events**.

The more useful early-warning comparison is the 2024 validation result:

- Persistence onset recall: **0%**
- Tuned LightGBM onset recall: **50%**

The 2025 test results were used only for reporting and were **not used to retune the model**.

---

## 12. Feature Importance

Gain-based feature importance showed that the final LightGBM relied mostly on recent HAB history.

| Feature | Gain % |
|---|---:|
| `last_log_cells` | **46.83%** |
| `weeks_since_bloom` | **22.04%** |
| `last_log_cells_chg1` | 8.14% |
| `season_cos` | 6.39% |
| `season_sin` | 6.09% |
| `n_habsos_samples` | 3.92% |
| `class_run_weeks` | 3.07% |
| `pos_weeks_12wk` | 1.68% |
| `last_class_prev` | 1.38% |
| `class_max_4wk` | 0.45% |
| `last_class` | 0.00% |
| `weeks_since_obs` | 0.00% |

### Main Finding

`last_log_cells` and `weeks_since_bloom` together account for about **69%** of total model gain.

The model relies mainly on:

- recent HAB magnitude,
- recent bloom history,
- recent changes in cell count,
- seasonality.

Feature importance shows how the model used the features. It does not prove that a feature causes a bloom.

---

## 13. Main Conclusions

### What worked

- Balanced class weighting improved LightGBM.
- Time-aware hyperparameter tuning produced the best LightGBM.
- The 12 core regional features performed better than the larger environmental feature sets.
- LightGBM improved 2024 Macro F1 and balanced accuracy over persistence.
- LightGBM caught **1 of 2** 2024 onset events.
- The final model had **0 severe bloom misses** in both the 2024 validation and 2025 test evaluations.

### Main challenge

The hardest part is predicting a **new onset** rather than predicting stable conditions.

Persistence is strong when HAB conditions stay similar from week to week.

LightGBM is more useful when the goal includes detecting possible changes, but that comes with more false alarms.

---

## 14. Important Limitations

### Few Onset Events

The 2024 validation set contains only **2 onset events**.

The 2025 test set contains **0 onset events**.

This means onset recall is based on very little evidence and must be interpreted carefully.

### Class Imbalance

The 2025 test set contains:

- 45 Class 0 weeks,
- only 1 Class 1 week,
- 6 Class 2 weeks.

This makes Macro F1 and Balanced Accuracy sensitive to a very small number of predictions.

### 2026 Holdout

The labeled 2026 holdout contains only Class 0.

Because of this, it cannot provide a meaningful three-class evaluation.

---

## 15. Saved Outputs

### Results

```text
results/experiment_results.csv
results/final_test_metrics.csv
results/final_test_class_metrics.csv
results/final_test_predictions.csv
results/feature_importance_gain.csv
results/advisor_priority_validation_metrics.csv
results/advisor_priority_test_metrics.csv
```

### Figures

```text
figures/final_confusion_matrix.png
figures/feature_importance_gain.png
```

### Final Model

```text
saved_model/final_lightgbm_model.pkl
saved_model/final_lightgbm_model.txt
```

The saved `.pkl` model was reloaded and tested.

```text
Predictions match: True
```

This confirms that the saved model reproduces the final test predictions.
