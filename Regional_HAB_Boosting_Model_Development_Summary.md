# Regional HAB Forecasting — Boosting Model Development Summary

## 1. Project Goal

The goal is to predict **next-week regional harmful algal bloom (HAB) severity**.

The final system has two parts:

1. **Tuned LightGBM** — predicts the next-week HAB class.
2. **CatBoost early-warning layer** — gives a more sensitive warning when HAB risk may be increasing.

A simple **persistence baseline** is also used for comparison.

---

## 2. Target Classes

The model predicts one of three classes based on cell count:

- **Class 0 — Low / No Bloom:** less than 5,000 cells/L
- **Class 1 — Elevated Risk:** 5,000 to 99,999 cells/L
- **Class 2 — Severe / Confirmed Bloom:** 100,000 cells/L or more

The prediction question is:

> **What HAB severity class will occur next week?**

---

## 3. Regional Dataset

Main modeling file:

```text
data/final/model_table_weekly_fe_reduced.parquet
```

Dataset size:

- **595 rows**
- **104 columns**
- **486 labeled rows**
- **109 unlabeled rows**

Time split:

- **Training:** through 2023
- **Validation:** 2024
- **Test:** 2025
- **Holdout:** 2026

Labeled rows by split:

- Train: **364**
- Validation: **52**
- Test: **52**
- Holdout: **18**

Training class counts:

- Class 0: **185**
- Class 1: **59**
- Class 2: **120**

2025 test class counts:

- Class 0: **45**
- Class 1: **1**
- Class 2: **6**

---

## 4. Core 12 Features

The final regional models use these 12 features:

```python
regional_core = [
    "last_class",
    "last_log_cells",
    "last_class_prev",
    "last_log_cells_chg1",
    "weeks_since_obs",
    "class_max_4wk",
    "n_habsos_samples",
    "weeks_since_bloom",
    "pos_weeks_12wk",
    "class_run_weeks",
    "season_sin",
    "season_cos"
]
```

These features mostly describe:

- recent HAB cell counts,
- recent bloom history,
- recent changes,
- persistence,
- seasonality.

---

## 5. Main Evaluation Metrics

### Accuracy
Percent of all predictions that are correct.

### Balanced Accuracy
Average recall across the three classes.

Useful because the classes are imbalanced.

### Macro F1
F1 score calculated equally across all three classes.

This prevents the common Class 0 from dominating the score.

### Onset Recall
Measures whether the model catches a new increase in HAB risk.

A true onset is:

```text
Previous class = 0
AND
Next week's true class = 1 or 2
```

Formula:

```text
Onset Recall = Onsets Caught / Total True Onsets
```

### Bloom Miss
A serious error:

```text
True Class 2 → Predicted Class 0
```

### False Alarm
A warning when the week is actually low risk:

```text
True Class 0 → Predicted Class 1 or 2
```

### Cost Scores
Used to penalize missed blooms more than false alarms.

```text
2:1 Cost = 2 × Bloom Misses + False Alarms
3:1 Cost = 3 × Bloom Misses + False Alarms
```

---

## 6. Persistence Baseline

The persistence baseline predicts:

> Next week will have the same class as the most recently observed class.

### 2024 Validation

- Accuracy: **0.846**
- Balanced Accuracy: **0.613**
- Macro F1: **0.622**
- Onset Recall: **0.00**
- Onsets Caught: **0 / 2**
- Bloom Misses: **0**
- False Alarms: **1**

This baseline is strong during stable HAB periods, but it does not provide useful early warning.

---

## 7. Tuned LightGBM

Final tuned configuration:

```python
LGBMClassifier(
    objective="multiclass",
    num_class=3,
    learning_rate=0.03,
    max_depth=5,
    n_estimators=100,
    num_leaves=15,
    class_weight="balanced",
    random_state=42,
    verbosity=-1
)
```

### 2024 Validation

- Accuracy: **0.865**
- Balanced Accuracy: **0.737**
- Macro F1: **0.717**
- Weighted F1: **0.872**
- Onset Recall: **0.50**
- Onsets Caught: **1 / 2**
- Bloom Misses: **0**
- False Alarms: **3**

Confusion matrix:

```text
               Pred 0   Pred 1   Pred 2
Actual 0          35        3        0
Actual 1           1        2        2
Actual 2           0        1        8
```

### Decision

**Selected as the primary three-class forecasting model.**

It gave the strongest overall 2024 validation performance.

---

## 8. Regularized LightGBM

A stronger-regularized LightGBM was tested to reduce overfitting.

### 2024 Validation

- Accuracy: **0.846**
- Balanced Accuracy: **0.670**
- Macro F1: **0.658**
- Onset Recall: **0.00**
- Bloom Misses: **0**
- False Alarms: **3**

### Decision

**Not selected.**

It lost the onset advantage and performed worse than the tuned LightGBM.

---

## 9. Early-Warning Weighted LightGBM

A custom class-weighting experiment was tested to make the model more sensitive to Classes 1 and 2.

Example class weights:

```python
{
    0: 1.0,
    1: 2.5,
    2: 2.0
}
```

### 2024 Validation

- Accuracy: **0.827**
- Balanced Accuracy: **0.719**
- Macro F1: **0.683**
- Onset Recall: **0.50**
- Bloom Misses: **0**
- False Alarms: **5**

### Decision

**Not selected.**

It caught the same number of onsets as the tuned LightGBM but produced more false alarms.

---

## 10. CatBoost

CatBoost was tested as another boosting model.

Configuration:

```python
CatBoostClassifier(
    loss_function="MultiClass",
    iterations=300,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=5,
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=False
)
```

### 2024 Validation

- Accuracy: **0.846**
- Balanced Accuracy: **0.700**
- Macro F1: **0.686**
- Onset Recall: **0.50**
- Onsets Caught: **1 / 2**
- Bloom Misses: **0**
- False Alarms: **3**

### Decision

CatBoost was weaker than tuned LightGBM as a normal three-class classifier, but it became useful as an **early-warning model** after threshold tuning.

---

## 11. XGBoost

XGBoost was trained using balanced sample weights.

### 2024 Validation

- Accuracy: **0.827**
- Balanced Accuracy: **0.661**
- Macro F1: **0.648**
- Onset Recall: **0.00**
- Bloom Misses: **0**
- False Alarms: **4**

### Decision

**Not selected.**

It did not improve early-warning performance or overall validation performance.

---

## 12. Two-Threshold LightGBM

This approach treated the three classes as ordered.

Two binary models were trained:

### Threshold A

```text
Class 0 vs Class 1 or 2
```

Question:

> Will next week become at-risk or bloom?

### Threshold B

```text
Class 0 or 1 vs Class 2
```

Question:

> Will next week be a severe bloom?

### 2024 Validation

- Accuracy: **0.846**
- Balanced Accuracy: **0.670**
- Macro F1: **0.658**
- Onset Recall: **0.00**
- Bloom Misses: **0**
- False Alarms: **3**

### Decision

**Not selected.**

It did not improve over the tuned multiclass LightGBM.

---

## 13. Main 2024 Model Comparison

| Model | Onset Recall | Bloom Misses | False Alarms | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.00 | 0 | 1 | 0.622 | 0.613 |
| **Tuned LightGBM** | **0.50** | **0** | 3 | **0.717** | **0.737** |
| Regularized LightGBM | 0.00 | 0 | 3 | 0.658 | 0.670 |
| CatBoost | 0.50 | 0 | 3 | 0.686 | 0.700 |
| XGBoost | 0.00 | 0 | 4 | 0.648 | 0.661 |
| Two-Threshold LightGBM | 0.00 | 0 | 3 | 0.658 | 0.670 |

### Main Conclusion

The tuned LightGBM was the best overall model for the main three-class forecast.

---

## 14. What Threshold Tuning Means

A model usually returns probabilities.

Example CatBoost output:

```text
P(Class 0) = 0.58
P(Class 1) = 0.27
P(Class 2) = 0.15
```

Normal prediction:

```text
Class 0
```

because Class 0 has the highest probability.

For early warning, we calculate:

```text
Risk Probability = P(Class 1) + P(Class 2)
```

Example:

```text
0.27 + 0.15 = 0.42
```

If the early-warning threshold is:

```text
0.35
```

then:

```text
0.42 >= 0.35
```

so the system raises an alert.

### Important Meaning

A **lower threshold**:

- catches more risky weeks,
- improves sensitivity,
- usually creates more false alarms.

A **higher threshold**:

- gives fewer warnings,
- reduces false alarms,
- may miss more onsets.

---

## 15. 2024 Threshold Tuning

### CatBoost

Fine threshold search showed:

| Threshold | Onset Recall | Onsets Caught | False Alarms | Macro F1 |
|---|---:|---:|---:|---:|
| 0.34 | 1.00 | 2/2 | 6 | 0.689 |
| 0.35 | 1.00 | 2/2 | 6 | 0.689 |
| 0.36 | 1.00 | 2/2 | 5 | 0.702 |
| 0.37 | 1.00 | 2/2 | 5 | 0.702 |
| 0.38 | 1.00 | 2/2 | 5 | 0.702 |
| 0.39 | 1.00 | 2/2 | 5 | 0.702 |
| 0.40 | 0.50 | 1/2 | 5 | 0.660 |

On the single 2024 validation set, **0.39** was the highest threshold that still caught both onsets.

However, 2024 had only **2 onset events**, so this was not enough evidence to lock 0.39 as the final threshold.

---

## 16. Historical CatBoost Backtesting

To make threshold selection more reliable, CatBoost was backtested across:

- 2020
- 2021
- 2022
- 2023
- 2024

A four-week gap was kept before each validation year.

Historical validation rows:

- **156**

Historical onset weeks:

- 2020: **6**
- 2021: **2**
- 2022: **1**
- 2023: **2**
- 2024: **2**

Total:

- **13 onset weeks**

### Historical Threshold Results

At threshold **0.35**:

- Onsets Caught: **6 / 13**
- Onset Recall: **0.462**
- Bloom Misses: **0**
- False Alarms: **33**
- Macro F1: **0.636**
- Balanced Accuracy: **0.659**

Lower thresholds increased onset recall but caused many more false alarms.

Examples:

| Threshold | Onset Recall | Onsets Caught | False Alarms |
|---|---:|---:|---:|
| 0.15 | 0.923 | 12/13 | 79 |
| 0.20 | 0.769 | 10/13 | 64 |
| 0.24 | 0.769 | 10/13 | 51 |
| 0.30 | 0.462 | 6/13 | 37 |
| 0.35 | 0.462 | 6/13 | 33 |

### Conclusion

Threshold tuning can improve onset detection, but the false-alarm trade-off becomes large.

Therefore:

> **0.35 is treated as a candidate early-warning operating threshold, not a proven universal threshold.**

---

## 17. Historical LightGBM Backtest

The tuned LightGBM was tested on the same historical validation folds.

### Results

- Validation Rows: **156**
- Onset Total: **13**
- Onsets Caught: **2**
- Missed Onsets: **11**
- Onset Recall: **0.154**
- Bloom Misses: **2**
- False Alarms: **23**
- Accuracy: **0.660**
- Balanced Accuracy: **0.583**
- Macro F1: **0.586**

### Historical Comparison

| Metric | Tuned LightGBM | CatBoost @ 0.35 |
|---|---:|---:|
| Onsets Caught | 2/13 | **6/13** |
| Onset Recall | 0.154 | **0.462** |
| Missed Onsets | 11 | **7** |
| Bloom Misses | 2 | **0** |
| False Alarms | **23** | 33 |
| Accuracy | 0.660 | 0.660 |
| Balanced Accuracy | 0.583 | **0.659** |
| Macro F1 | 0.586 | **0.636** |

### Conclusion

CatBoost is more sensitive to new risk, but it produces more false alarms.

This supports using:

- **LightGBM for the main forecast**
- **CatBoost for early warning**

---

## 18. Final 2025 Test

After model design was locked, the models were retrained using all labeled data through 2024.

Final training rows:

- **416**

The 2025 test set was then used only for final evaluation.

---

## 19. Final LightGBM — 2025

### Results

- Accuracy: **0.904**
- Balanced Accuracy: **0.589**
- Macro F1: **0.600**
- Weighted F1: **0.932**
- Bloom Misses: **0**
- False Alarms: **3**

There were **0 true onset events** in 2025.

Therefore:

```text
Onset Recall = NaN / undefined
```

This is correct because onset recall cannot be calculated when there are no true onsets.

---

## 20. CatBoost Early Warning — 2025

Final candidate threshold:

```text
0.35
```

### Results

- Accuracy: **0.904**
- Balanced Accuracy: **0.637**
- Macro F1: **0.626**
- Weighted F1: **0.932**
- Bloom Misses: **0**
- False Alarms: **4**

Confusion matrix:

```text
               Pred 0   Pred 1   Pred 2
Actual 0          41        4        0
Actual 1           0        0        1
Actual 2           0        0        6
```

Important findings:

- **All 6 severe Class 2 bloom weeks were correctly identified.**
- There were **0 severe bloom misses**.
- The 4 false alarms were Class 0 weeks predicted as Class 1.
- The single Class 1 week was conservatively predicted as Class 2.
- There were no true onset events in 2025, so onset recall could not be tested.

---

## 21. Final System

### Primary Model — LightGBM

Purpose:

> Predict the next-week HAB severity class.

Output example:

```text
Predicted Next-Week HAB Class: Class 2
```

### Early-Warning Model — CatBoost

Purpose:

> Detect elevated combined probability of Class 1 or Class 2.

Calculation:

```text
Risk Probability = P(Class 1) + P(Class 2)
```

Candidate threshold:

```text
0.35
```

Output example:

```text
HAB Early Warning: YES
Risk Probability: 42%
Warning Level: Class 1
```

The warning level is:

- Class 1 if `P(Class 1) > P(Class 2)`
- Class 2 if `P(Class 2) > P(Class 1)`
- Class 0 when no warning is triggered

---

## 22. Feature Importance

Final LightGBM gain importance:

| Feature | Importance % |
|---|---:|
| `last_log_cells` | **59.69%** |
| `weeks_since_bloom` | **12.27%** |
| `season_cos` | **7.04%** |
| `last_log_cells_chg1` | **6.53%** |
| `season_sin` | **4.92%** |
| `class_run_weeks` | 3.12% |
| `last_class_prev` | 2.80% |
| `n_habsos_samples` | 2.38% |
| `pos_weeks_12wk` | 1.01% |
| `class_max_4wk` | 0.24% |
| `last_class` | ~0% |
| `weeks_since_obs` | 0% |

### Interpretation

The model mostly depends on:

1. recent cell concentration,
2. how recently a bloom occurred,
3. recent change in cell counts,
4. seasonality.

The strongest feature is:

```text
last_log_cells
```

This means the most recent observed HAB cell concentration is the strongest signal for next-week severity.

---

## 23. Environmental Feature Experiments

Environmental groups tested:

- temperature,
- chlorophyll,
- wind,
- nutrients,
- rainfall.

Results:

| Feature Set | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| **Core 12** | **0.865** | **0.737** | **0.717** |
| Core + Temperature | 0.808 | 0.624 | 0.615 |
| Core + Chlorophyll | 0.769 | 0.665 | 0.638 |
| Core + Wind | 0.808 | 0.710 | 0.670 |
| Core + Nutrients | 0.827 | 0.661 | 0.648 |
| Core + Rain | 0.827 | 0.691 | 0.673 |
| Core + Major Environmental Features | 0.788 | 0.615 | 0.604 |

### Conclusion

The **Core 12** features performed best.

This does **not** mean environmental factors are biologically unimportant.

It means that in this dataset and regional next-week setup, the tested environmental variables did not improve prediction beyond recent HAB history and seasonality.

Possible reasons:

- environmental effects may occur with different time lags,
- regional aggregation may hide local effects,
- some variables may be noisy or sparse,
- recent cell counts may already summarize the current bloom state.

---

## 24. Main Limitations

### Rare Onset Events

Onset events are uncommon.

- 2024 validation: **2 onsets**
- Historical 2020–2024 backtest: **13 onsets**
- 2025 test: **0 onsets**

This makes onset recall unstable.

### Class Imbalance

Class 1 is much less common than Class 0.

This makes the intermediate-risk class harder to learn.

### Strong Persistence

Recent HAB conditions are very predictive.

This makes stable weeks easier to predict.

The harder problem is:

> **Predicting when the system changes into a new bloom or higher-risk state.**

### Early-Warning False Alarms

Lower thresholds catch more onsets but can produce many false alarms.

### Environmental Variables

Environmental features did not improve the current regional validation results.

Future work should test:

- different time lags,
- better spatial alignment,
- site-level or subregional models.

---

## 25. Final Boosting Model Decision

### Primary Forecast

**Tuned LightGBM**

Why:

- best overall 2024 multiclass validation performance,
- strong Macro F1,
- strong balanced accuracy,
- zero 2024 bloom misses,
- zero 2025 bloom misses.

### Early Warning

**CatBoost + candidate 0.35 threshold**

Why:

- more sensitive to historical onset events,
- caught more historical onsets than LightGBM,
- zero historical bloom misses at the candidate threshold,
- correctly detected all six Class 2 weeks in 2025.

Main trade-off:

> CatBoost provides better sensitivity but more false alarms.

---

## 26. Models Not Selected

### Regularized LightGBM
Did not improve onset recall or overall validation performance.

### Early-Warning Weighted LightGBM
Same onset recall as tuned LightGBM but more false alarms.

### XGBoost
Lower performance and no onset detection improvement.

### Two-Threshold LightGBM
Did not improve over the standard tuned LightGBM.

---

## 27. Saved Final Models

Final artifacts are stored under:

```text
models/boosting_mohammed/
```

Recommended structure:

```text
models/
└── boosting_mohammed/
    ├── saved_model/
    │   ├── tuned_lightgbm_final_through_2024.pkl
    │   ├── catboost_early_warning_final_through_2024.cbm
    │   └── regional_hab_system_config.json
    │
    ├── figures/
    │   ├── lightgbm_feature_importance.png
    │   ├── environmental_feature_comparison.png
    │   ├── catboost_2025_confusion_matrix.png
    │   └── final_model_comparison.png
    │
    └── results/
        ├── Main_final_model_summary.csv
        ├── Main_environmental_results.csv
        └── Main_historical_threshold_results.csv
```

---

## 28. Saved System Verification

The saved models were reloaded successfully.

Example verification output:

```text
All final artifacts loaded successfully.
Early-warning threshold: 0.35

Predicted Next-Week HAB Class: Class 2

HAB Early Warning: YES
Risk Probability: 96%
Warning Level: Class 2
```

This confirms that:

- the saved LightGBM model loads correctly,
- the saved CatBoost model loads correctly,
- the threshold config loads correctly,
- both final prediction outputs work.

---

## 29. Final Takeaways

1. **LightGBM is the main next-week severity model.**
2. **CatBoost is used as a more sensitive early-warning layer.**
3. **Recent HAB history is the strongest predictive signal.**
4. **Environmental feature groups did not improve the current regional model.**
5. **The hardest problem is predicting new bloom onset, not stable conditions.**
6. **Lower warning thresholds improve sensitivity but increase false alarms.**
7. **2025 had no onset events, so onset recall could not be evaluated in the final test year.**
8. **Both final models produced zero severe bloom misses in 2025.**

---

## 30. Simple Final System Flow

```text
Regional weekly HAB data
        |
        v
12 core regional features
        |
        +------------------------------+
        |                              |
        v                              v
Tuned LightGBM                    CatBoost
        |                              |
        v                              v
Class 0 / 1 / 2              P(Class 1) + P(Class 2)
                                       |
                                       v
                              Compare with 0.35
                                       |
                                       v
                              Early Warning YES / NO
                              Risk Probability
                              Warning Level
```

