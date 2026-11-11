# Random Forest Model — Regional and Site-Level HAB Forecasting

**Notebook:** `notebooks/random_forest.ipynb`
**Branch:** `random-forest-model`

## Overview

This notebook develops a **random forest model** to predict next-week harmful algal bloom (HAB) severity along the Sarasota focus coast.

The model uses information from week **t** to predict HAB severity during week **t+1**.

The model predicts one of three classes:

- **Class 0:** < 5,000 cells/L
- **Class 1:** 5,000–99,999 cells/L
- **Class 2:** >= 100,000 cells/L

Following the client's feedback, the forecast is also used as an **early-warning alert**: "next week will be at risk or bloom (class 1 or 2)". Two versions were built:

- **Regional model (main product):** one forecast per week for the whole focus coast
- **Site-level model (secondary):** one forecast per sampling site per week

---

## 1. Datasets

| Model | Dataset | Rows | Labeled rows |
|---|---|---:|---:|
| Regional | `data/final/model_table_weekly_fe_reduced.parquet` | 595 weeks | 486 |
| Site-level | `data/final/site_week_table_fe.parquet` | 21,718 site-weeks | 16,037 |

Rows where the next week wasn't sampled have no label and were dropped.

### Time-Based Split

| Split | Target Years | Regional rows | Site-level rows |
|---|---|---:|---:|
| Train | 2015–2023 | 364 | 12,379 |
| Validation | 2024 | 52 | 1,653 |
| Test | 2025 | 52 | 1,455 |
| Holdout | 2026 | 18 | 550 |

A random split was not used because this is a forecasting problem.

**Class balance changes over time.** Blooms are 21% of training site-weeks but only 5% of 2025, and 2026 has no blooms at all, so it can only measure false alarms.

---

## 2. Final 12 Core Features

**Regional model** (same 12 as the LightGBM model):

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

**Site-level model:**

1. `site_last_class`: class at the site's most recent sample
2. `site_last_log_cells`: cell count at that sample (log10)
3. `site_weeks_since_obs`: weeks since the site was last sampled
4. `nbr_class`: highest class this week at other sites within 20 km
5. `reg_last_class`: most recent class for the whole focus coast
6. `reg_class_max_4wk`: highest regional class over the last 4 weeks
7. `reg_n_samples`: number of samples along the coast that week
8. `weeks_since_bloom`
9. `pos_weeks_12wk`
10. `class_run_weeks`
11. `season_sin`
12. `season_cos`

scikit-learn 1.4+ lets the random forest handle missing values directly, so missing values were not filled in. They're missing for a reason, for example a site's first sample has no "last class", not at random.

**Caution:** `n_habsos_samples` / `reg_n_samples` measure sampling effort. Agencies sample more when they suspect a bloom, so these may reflect people reacting to a bloom rather than the environment causing one.

---

## 3. Evaluation Approach

### Metrics

Every model is scored with the same report card (`early_warning_report()`):

| Metric | Meaning |
|---|---|
| Accuracy | Share of weeks predicted in the right class |
| Balanced Accuracy | Average recall across the three classes |
| Macro F1 | Average F1 across the three classes |
| **Onset Recall** | Of the **new** blooms (no bloom last time, at risk or bloom next week), the share that got an alert |
| **Bloom Misses** | Real bloom weeks (class 2) with no alert |
| **False Alarms** | Alerts in weeks that turned out to be class 0 |
| Misses | All risky weeks (class 1 or 2) with no alert |
| **Cost (3:1)** | 3 × misses + false alarms. Lower is better. 2:1 is also reported |

The client asked for onset recall as a primary metric and for misses to cost more than false alarms (2:1 or 3:1). So **models are chosen by onset recall and the 3:1 cost, not by accuracy or macro F1.** Most weeks have no bloom, so accuracy mostly rewards staying quiet.

**Why 3:1 is the main ratio:** at 2:1, the cheapest regional threshold catches almost no new blooms, which works against the client's early-warning priority.

### Rolling Validation

2024 alone is a small test: 52 regional weeks. Instead, each validation year is predicted by a model trained only on earlier years, and the results are added together:

| Validate on | Train on |
|---|---|
| 2019 | 2015–2018 |
| 2020 | 2015–2019 |
| 2021 | 2015–2020 |
| 2024 | 2015–2023 |

2022 and 2023 are skipped as validation years because they have almost no labels (8 and 17 regional weeks). They're still used for training. 2025 was kept untouched until the final test.

**How much results move by chance:** the same forest was rerun with 4 random seeds. A change only counts as real if it moves the 3:1 cost by more than **twice** the seed-to-seed range: **6** regionally and **50** at site level.

---

## 4. First Random Forest Model

The first random forest was a site-level model with the 12 site-level core features, trained on 2015–2023 and scored on 2024:

```python
RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=10, random_state=42)
```

It was compared with a **persistence baseline**: "next week = last week".

### Initial Site-Level Model — 2024 Validation

| Model | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Bloom Misses | False Alarms | Cost 3:1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | **0.887** | 0.478 | **0.490** | 0.00 (0 / 90) | 19 | **64** | 334 |
| Random forest | 0.850 | **0.499** | 0.472 | **0.51 (46 / 90)** | **14** | 133 | **265** |

On accuracy and macro F1 the two look similar. On early warning, the forest is much better: it catches about half of the new blooms, which the baseline can't catch at all, though with about twice the false alarms.

---

## 5. Handling Class Imbalance and the Alert Threshold

Two settings were compared with rolling validation:

- **`class_weight`:** `"balanced"` (counts rare bloom weeks more) vs none
- **`min_samples_leaf`:** 10 vs 30 (larger leaves memorize less)

On top of these, the forest raises an alert when its **probability of class 1 or 2 is above a threshold**. The threshold is the main dial between catching blooms and raising false alarms.

**Threshold rule:** take the threshold with the lowest 3:1 cost. If other thresholds are within 5% of that cost, take the one that catches the most new blooms.

### Site-Level Results (rolling validation)

| Model | Threshold | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Bloom Misses | False Alarms | Cost 3:1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | - | **0.896** | 0.524 | 0.518 | 0.00 (0 / 291) | 74 | **302** | 1175 |
| Balanced, leaf 10 | 0.50 | 0.871 | **0.621** | 0.530 | **0.55** (161 / 291) | **31** | 546 | **960** |
| **Balanced, leaf 30 (chosen)** | **0.60** | 0.874 | 0.619 | **0.532** | 0.52 (151 / 291) | 33 | 520 | 973 |
| No weights, leaf 10 | 0.20 | 0.859 | 0.602 | 0.495 | 0.51 (149 / 291) | 33 | 587 | 1037 |
| No weights, leaf 30 | 0.20 | 0.863 | 0.600 | 0.495 | 0.52 (152 / 291) | 32 | 561 | 999 |

### Decision

The four forest versions are within noise of each other, and all clearly beat the baseline. **Balanced class weights with `min_samples_leaf=30` were kept**, because larger leaves are less likely to memorize. The regional model uses balanced weights with `min_samples_leaf=10`, since it has far fewer rows.

**The threshold trade-off** (site level; the full table is in the notebook):

| Threshold | Onset Recall | False Alarms |
|---:|---:|---:|
| 0.40 | 0.72 | 855 |
| 0.60 (chosen) | 0.52 | 520 |
| 0.70 | 0.43 | 382 |

This is the table to show the client so they can choose their operating point.

---

## 6. Environmental Feature Experiments

Feature groups were added to the core features one at a time (and a few core features were dropped) to see whether they improved prediction.

Groups tested:

- Temperature
- Chlorophyll
- Wind
- Bay nutrients (regional only)
- Rain

**Fair comparison:** chlorophyll data starts in 2018 and wind in 2019. For those groups, **both** the core model and the core-plus-group model were trained only on years with data. Otherwise the test would measure "less training data", not the feature.

The **Cost** columns are the lowest 3:1 cost over the rolling validation years (negative change = better). The other scores are at each version's chosen threshold.

### Regional Results (a change has to beat 6 to count)

| Feature Set | Trained From | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Cost 3:1 | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| Core 12 | 2015 | 0.732 | 0.705 | 0.678 | 0.43 | 53 | - |
| Core − `weeks_since_bloom` | 2015 | 0.798 | 0.697 | 0.698 | 0.07 | 51 | −2 |
| Core − `last_class` | 2015 | 0.776 | 0.694 | 0.688 | 0.14 | 53 | 0 |
| Core + Temperature | 2015 | 0.803 | 0.700 | 0.702 | 0.07 | 50 | −3 |
| Core + Nutrients | 2015 | 0.760 | 0.694 | 0.682 | 0.21 | 52 | −1 |
| Core + Rain | 2015 | 0.787 | 0.691 | 0.690 | 0.07 | 52 | −1 |
| Core 12 (2018+ years) | 2018 | 0.683 | 0.630 | 0.619 | 0.36 | 68 | - |
| Core + Chlorophyll | 2018 | 0.623 | 0.608 | 0.568 | 0.43 | 77 | +9 |
| Core 12 (2019+ years) | 2019 | 0.710 | 0.633 | 0.607 | 0.20 | 50 | - |
| Core + Wind | 2019 | 0.718 | 0.648 | 0.614 | 0.20 | 47 | −3 |

### Site-Level Results (a change has to beat 50 to count)

| Feature Set | Trained From | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Cost 3:1 | Change |
|---|---|---:|---:|---:|---:|---:|---:|
| Core 12 | 2015 | 0.874 | 0.619 | 0.532 | 0.52 | 934 | - |
| Core − `weeks_since_bloom` | 2015 | 0.890 | 0.623 | 0.552 | 0.47 | 910 | −24 |
| Core − `reg_last_class` | 2015 | 0.881 | 0.636 | 0.549 | 0.55 | 903 | −31 |
| Core − `site_last_class` | 2015 | 0.864 | 0.621 | 0.521 | 0.59 | 930 | −4 |
| Core + Temperature | 2015 | 0.875 | 0.633 | 0.544 | 0.57 | 923 | −11 |
| Core + Rain | 2015 | 0.871 | 0.631 | 0.534 | 0.54 | 941 | +7 |
| Core 12 (2018+ years) | 2018 | 0.887 | 0.596 | 0.545 | 0.40 | 1002 | - |
| Core + Chlorophyll | 2018 | 0.881 | 0.597 | 0.546 | 0.42 | 996 | −6 |
| Core 12 (2019+ years) | 2019 | 0.848 | 0.543 | 0.400 | 0.59 | 667 | - |
| Core + Wind | 2019 | 0.858 | 0.573 | 0.464 | 0.58 | 664 | −3 |

### Decision

None of the environmental feature groups, and none of the drops, improved validation by more than chance. **Both final models kept the 12 core features.**

A few regional versions (for example + Temperature) look slightly better on accuracy and macro F1 but catch far fewer new blooms (onset recall 0.07 vs 0.43). Because onset recall is the primary metric, those aren't improvements.

Following the client's guidance, environmental variables were not forced in. This does not mean they don't matter for HAB formation. It only means they did not improve this next-week forecast with the current data.

**Two side findings:**

- Dropping `site_last_class` barely changes the site model. The forest gets the same "last week" information from `site_last_log_cells`.
- Permutation importance (section 8) flagged `season_cos`, `pos_weeks_12wk`, `reg_n_samples` and `class_run_weeks`. Retraining without each one, or all four, lowered cost by only 25–41, which is within noise, so they stay.

---

## 7. Final Models — Validation Results

### Regional Model (main product)

Three ways to make the regional forecast were compared. Across the four validation years there are only **14 regional new blooms**, so a difference of one or two weeks isn't meaningful.

| Model | Threshold | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Bloom Misses | False Alarms | Cost 3:1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline (next week = last week) | - | **0.792** | 0.694 | **0.695** | 0.00 (0 / 14) | 2 | **13** | 55 |
| **Regional random forest** | 0.40 | 0.732 | **0.705** | 0.678 | **0.43 (6 / 14)** | **1** | 29 | **53** |
| Site forest combined (highest site probability each week) | 0.25 | 0.721 | 0.686 | 0.673 | 0.36 (5 / 14) | **0** | 29 | 56 |

- **At the regional level, the forest does not clearly beat persistence overall.** Cost is within noise, and accuracy and macro F1 are close.
- **What it adds is early warning:** 6 of 14 new blooms caught, at about 4 extra false alarms per year.
- The likely limit is the small regional dataset: 364 training weeks, compared with 12,000+ site-weeks.

**Note for comparison with LightGBM:** these numbers are pooled over four validation years at the chosen alert threshold. The LightGBM summary reports 2024 only, with the most likely class, so the numbers are not directly comparable.

### Site-Level Model (secondary)

Chosen: balanced weights, `min_samples_leaf=30`, threshold 0.60 (see section 5). It catches **about half of all new blooms** (151 / 291) and cuts bloom misses from 74 to 33, at the cost of more false alarms (520 vs 302).

---

## 8. Feature Importance

Permutation importance asks: **"If I scramble one feature, how much worse does the model get?"** The score is the lowest 3:1 cost over all four validation years, with the threshold picked again after each scramble. Positive = the cost goes up = the forest relies on that feature.

### Site-Level Model

| Feature | Cost Increase | Spread | Reading |
|---|---:|---:|---|
| `nbr_class` | +184 | 21 | **By far the most important:** highest class at nearby sites this week |
| `site_last_log_cells` | +38 | 14 | The site's last cell count |
| `weeks_since_bloom` | +15 | 12 | Small; the spread is nearly as large |
| All others | −28 to +3 | | No clear help |

The forest's early-warning skill comes mainly from **"is a nearby site already showing cells?"** Blooms spread along the coast, so neighbors give the warning.

An earlier check (2024 only, scored on macro F1) ranked `pos_weeks_12wk` and `season_cos` highest. That ranking changes once the model is scored on early warning over several years.

### Regional Model

Every feature is within ±4 cost. With only 14 new blooms, no single feature stands out.

**Caution:** this shows how the model predicts, not what causes blooms. Similar features, like `site_last_class` and `site_last_log_cells`, also share credit.

---

## 9. Site-Level Reliability

The client asked for site-level prediction to stay secondary until it's shown to be reliable across sites. Results split into thirds of the coast by site latitude:

| Part of Coast | Sites | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Bloom Misses | False Alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| South | 46 | 0.872 | 0.613 | 0.566 | 0.49 (62 / 127) | 25 | 176 |
| Central | 45 | 0.860 | 0.579 | 0.493 | 0.58 (54 / 93) | 8 | 182 |
| North | 45 | 0.888 | 0.665 | 0.498 | 0.49 (35 / 71) | 0 | 162 |

- Onset recall is **similar along the coast** (0.49–0.58).
- **Most bloom misses are in the south** (25 of 33).
- Only **21 of 136 sites** have 5+ new blooms. At those, onset recall ranges from **0.17 to 0.80** (median 0.40).

**Reliability at individual sites hasn't been shown yet**, so site-level stays secondary. Every site in the table is marked offshore, so an inshore/offshore comparison isn't possible.

---

## 10. Final Test — 2025

All settings, features and thresholds were fixed by the rules above before this step. Each model was retrained on all years up to 2024 and scored once on 2025.

| Model | Accuracy | Balanced Acc. | Macro F1 | Onset Recall | Bloom Misses | False Alarms | Cost 3:1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Site: baseline | **0.907** | 0.499 | **0.463** | 0.00 (0 / 33) | 11 | **81** | 180 |
| Site: random forest | 0.869 | **0.592** | 0.444 | **0.85 (28 / 33)** | **3** | 156 | **171** |
| Regional: baseline | **0.962** | **0.659** | **0.637** | n/a (0 new blooms) | 0 | **1** | **1** |
| Regional: random forest | 0.904 | 0.637 | 0.626 | n/a | 0 | 4 | 4 |
| Regional: site forest combined | 0.827 | 0.463 | 0.514 | n/a | 0 | 5 | 5 |

- **Site level:** the forest caught **28 of 33 new blooms** and cut bloom misses from 11 to 3, with about twice the false alarms. It wins on 3:1 cost.
- **Regional:** 2025 had **no regional new blooms**. All 7 risky weeks followed risky weeks, so persistence was nearly perfect, and **2025 can't measure regional early warning**. The forest raised 3 extra false alarms over the year.
- Note: the 2025 cell was first run by mistake before the feature tests were finished. Nothing was changed because of it, and the core features were kept anyway, so these numbers are unaffected.

---

## 11. Conclusion

> Scored on early warning with misses costing 3× a false alarm, the site-level random forest catches about half of new bloom onsets in rolling validation (and 28 of 33 in 2025), where the persistence baseline catches none. Its main signal is bloom activity at neighboring sites. At the regional level, the forest is roughly tied with persistence overall but catches 6 of 14 new blooms the baseline misses, for about 4 extra false alarms a year. No environmental feature group improved validation by more than chance, so the final models use bloom-history and season features only.

### Next Steps

1. **Choose the operating point with the client:** show the threshold trade-off tables and confirm 3:1.
2. **Get more regional signal:** 14 regional new blooms is too few to separate models. Try other regional labels (`y_class_box`, `y_class_mote_25km`) or a regional feature built from neighbor activity.
3. **Site reliability:** look at the south and at sites with low onset recall.
4. **Compare with LightGBM on the same footing:** same rolling validation years and the same early-warning report card.
5. **Sampling effort:** check whether some sites are only sampled when a bloom is suspected.

---

## Appendix A. Setup (local, VS Code)

- Installed `pyarrow` (for `.parquet` files) and `scikit-learn` into the project's `.venv` (pyarrow 26.0.0, scikit-learn 1.9.1).
- `scikit-learn` is not in `requirements.txt` yet and should be added for local users.
- Colab users need to change the `../data/final/...` paths to wherever the data lives in their session or Drive.

## Appendix B. Missing Data

| Feature group | Missing in training | Missing in 2024–2025 | Why |
|---|---|---|---|
| Core features | Under 10% | ~0% | Start of the record, or a site's first sample |
| Wind (`uwnd_*`) | 60% | 0% | Data starts in 2019 |
| Chlorophyll | 43% | 0% | Satellite data starts in 2018 |
| Rain, air temperature | ~0% | 0% (90–100% missing in 2026) | |

Because wind and chlorophyll are missing by year, a model could learn "missing chlorophyll means an older year". That's why those groups were tested only on years where they exist (section 6).

## Appendix C. Notebook Guide

| Section | What it does |
|---|---|
| Part 1 (cells 1–13) | Loads data, baseline, first random forest, macro-F1 permutation importance (2024) |
| Early-warning report card | `early_warning_report()` and `to_class()`. Rescores the Part 1 models |
| Rolling validation | `rolling_predictions()`, `threshold_sweep()`, `pick_threshold()`, `best_cost()` |
| Site-level model | Compares settings, checks seed noise, threshold trade-off table |
| Feature tests (site) | Drops and environmental groups, final site feature set |
| Site reliability | Results by part of the coast and per site |
| Regional model | Baseline vs regional forest vs combined site forest |
| Feature tests (regional) | Same tests for the regional model |
| Permutation importance | Cost-based, for the final site and regional models |
| Final check on 2025 | One-time test of the chosen models |
