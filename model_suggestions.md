# HAB Risk Modeling: Proposed Models

**Project:** Forecasting Harmful Algal Bloom (HAB) Risk, Biointerphase × Break Through Tech AI Studio (Fall 2026)
**Milestone:** October, Model Development

## The problem these models need to handle

- **Target:** 3 ordered classes from _K. brevis_ cell counts: `0 = no bloom` (< 5,000 cells/L), `1 = at risk` (5,000–99,999), `2 = bloom` (≥ 100,000). Label = `y_class`, next week's max class.
- **Class imbalance:** class 0 is most common, and class 1 is the rarest (regional labelled weeks: 286 / 65 / 135)
- **Persistence:** EDA shows HAB status carries over strongly from week to week
- **Interpretability:** success criteria require explaining the top environmental drivers, not just predicting
- **Data type:** tabular, multi-source (water quality, weather, satellite), time-ordered

## Summary

| #   | Model                                     | Type                     | Main reason it's included                          | Library                       |
| --- | ----------------------------------------- | ------------------------ | -------------------------------------------------- | ----------------------------- |
| 1   | Persistence                               | Rule-based baseline      | Minimum bar any ML model must beat                 | `pandas`                      |
| 2   | Ordinal / Multinomial Logistic Regression | Linear, interpretable    | Respects class order; explainable coefficients     | `statsmodels`, `scikit-learn` |
| 3   | Random Forest                             | Tree ensemble (bagging)  | Nonlinear effects + interactions, robust           | `scikit-learn`                |
| 4   | Gradient Boosting                         | Tree ensemble (boosting) | Usually strongest on tabular data                  | `lightgbm` / `scikit-learn`   |
| 5   | GAM                                       | Additive, interpretable  | Smooth nonlinear curves that are still explainable | `pygam`                       |

**Run order:** Persistence → Logistic → Random Forest → Gradient Boosting → GAM

The order runs from simplest to most complex. Each step has to justify its added complexity by beating the one before it.

---

## 1. Persistence Baseline

**What it is:** Predict that next week's class equals this week's class at the same site. Nothing is learned.

**Why I suggested it:**

- Our EDA shows strong week-to-week bloom persistence, so this rule will already score well.
- It is the minimum bar. If an ML model can't beat "copy last week," it adds no value.
- It is standard practice in time-dependent ecological forecasting.

**Watch out for:** HABSOS sampling is irregular. Aggregate to a site × week grid before shifting labels.

_What this means:_ "last week" only makes sense if every site has one row per week. HABSOS samples are taken on uneven dates, so a naive "previous row" could be 2 days or 3 months earlier.

Example of raw data at one site:

| date   | site | class |
| ------ | ---- | ----- |
| Jun 2  | A    | 0     |
| Jun 4  | A    | 1     |
| Jun 30 | A    | 2     |

A raw `shift(1)` says the class before Jun 30 was 1. That reading is from 26 days earlier, not last week.

After building a site × week grid:

| week    | site | class           |
| ------- | ---- | --------------- |
| Jun wk1 | A    | 1 (max of 0, 1) |
| Jun wk2 | A    | _missing_       |
| Jun wk3 | A    | _missing_       |
| Jun wk4 | A    | 2               |

Now `shift(1)` really means "one week earlier," and gaps are visible instead of hidden.

_Our pipeline already does this_ (`03_model_table.ipynb`):

| Step                    | Where it happens                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| Define a site           | `site_key` = lat/lon rounded to 2 decimals (~1 km); 136 sites with ≥ 20 samples                               |
| Bin into weeks          | `week_start` = the Monday of each week                                                                        |
| One row per site-week   | max class of that week                                                                                        |
| Every site × every week | full grid, then rows kept if they have a label or a sample                                                    |
| "Last week's class"     | `last_class` / `site_last_class` = last **observed** class carried forward, with its age in `weeks_since_obs` |

So **the persistence baseline is just `last_class` as the prediction** (`site_last_class` for the site-week table). Notebook 05 already scored it on the regional table: AUC 0.880, and onset AUC 0.500, because it can never predict a new bloom.

_Assumptions checked against notebooks 02–05:_

| Assumption                                      | What the notebooks show                                                                         | Status                                                                                                                   |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Sampling is irregular                           | 82% of weeks have a focus-coast sample; 2022–2023 nearly empty (~180 samples/yr vs 3,000–5,000) | ✅ confirmed                                                                                                             |
| No stable site ID                               | sites built by rounding lat/lon (`site_key`)                                                    | ✅ confirmed                                                                                                             |
| Classes come from cell counts                   | `pd.cut(CELLCOUNT)` at 5k / 100k (`hab_class_10k` keeps the old 10k cutoff)                     | ✅ confirmed                                                                                                             |
| Unsampled week ≠ no bloom                       | `y_class` is NaN when next week is unsampled; only the _feature_ is carried forward             | ✅ handled                                                                                                               |
| Some sites only sampled during suspected blooms | not checked yet                                                                                 | ❓ open. If true, sampled weeks over-represent blooms. Compare `n_habsos_samples` in bloom vs non-bloom weeks to test it |

**Read more:**

- [Forecasting: Principles and Practice, Simple forecasting methods (naïve method)](https://otexts.com/fpp3/simple-methods.html), Hyndman & Athanasopoulos

---

## 2. Ordinal Logistic Regression (fallback: Multinomial Logistic)

**What it is:** Combines features into a weighted score and learns cutoffs that split the score into the ordered classes 0 < 1 < 2.

**Why I suggested it:**

- Our target is naturally ordered, and this model uses that order instead of treating the classes as unrelated.
- Coefficients become odds ratios (e.g., "+1°C raises the odds of a higher bloom class by X%"), which directly supports the "top 5 drivers" success metric.
- It is a transparent, well-understood baseline that is common in environmental risk modeling.

**Watch out for:**

- It assumes each feature affects every cutoff the same way (the proportional odds assumption).
- If that assumption fails, switch to multinomial logistic with `class_weight='balanced'`.
- Scale features before fitting.

**Read more:**

- [statsmodels: Ordinal Regression example](https://www.statsmodels.org/stable/examples/notebooks/generated/ordinal_regression.html)
- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- _An Introduction to Statistical Learning_ (ISLP), Ch. 4 Classification, free at [statlearning.com](https://www.statlearning.com/)

---

## 3. Random Forest

**What it is:** Hundreds of decision trees, each trained on a random sample of rows and features, that vote on the final class.

**Why I suggested it:**

- It captures nonlinear effects and interactions (e.g., high temperature _and_ low wind) that logistic regression misses.
- It handles mixed, skewed environmental variables and outliers well with little preprocessing.
- `class_weight='balanced'` helps with our class imbalance.
- It is a common, reliable choice in ecological prediction.

**Watch out for:** Built-in `feature_importances_` is biased toward continuous and high-variance features. Use permutation importance for driver analysis.

**Read more:**

- [scikit-learn: Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- ISLP, Ch. 8 Tree-Based Methods ([statlearning.com](https://www.statlearning.com/))

---

## 4. Gradient Boosted Trees (LightGBM / HistGradientBoosting)

**What it is:** Trees built one after another, where each new tree corrects the errors of the ones before it.

**Why I suggested it:**

- It is usually the top-performing model family on tabular data like ours.
- It picks up threshold effects (e.g., risk jumps above a temperature or chlorophyll level) and feature interactions.
- It handles missing values natively, which matters with our multi-source gaps.
- SHAP values give per-prediction explanations of which features drove the risk up or down.

**Watch out for:** It overfits easily. Use early stopping on a time-based validation set, and start with shallow trees.

- In notebook 05, a small LightGBM scored **lower** than logistic regression on the regional table (only 364 training weeks). Try it on the site-week table (12,379 training rows) instead.

**Read more:**

- [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/)
- [scikit-learn: Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
- [SHAP documentation](https://shap.readthedocs.io/en/latest/)

---

## 5. GAM (Generalized Additive Model)

**What it is:** Like logistic regression, but each feature gets a smooth curve instead of a straight line.

**Why I suggested it:**

- Ecological processes are rarely linear (e.g., risk stays flat until a temperature threshold, then rises).
- It shows those curves directly, so it balances accuracy and interpretability.
- It is widely used in marine and ecological modeling.
- Its curves can validate the drivers found by the tree models.

**Watch out for:** `pygam.LogisticGAM` is **binary only**. Use our `y_binary` label (1 = at risk or bloom), or treat it as optional if time is short.

**Read more:**

- [pyGAM documentation](https://pygam.readthedocs.io/en/latest/)
- ISLP, Ch. 7 Moving Beyond Linearity ([statlearning.com](https://www.statlearning.com/))

---

## Evaluation Plan (applies to all models)

1. **Split by time, not randomly.** Already done: the `split` column = train (target year ≤ 2023), validate (2024), test (2025), holdout (2026). A random split leaks future information.
2. **Predict ahead.** Features at week _t_ predict the class at week _t + 1_ (`y_class`). So our current advance warning is 1 week.
3. **Test with and without last week's class as a feature.**
   - With it, the model mostly learns persistence.
   - Without it, the model shows which environmental drivers matter.
   - Start from `regional_core` / `site_core` in `feature_sets.json`, then add one candidate group at a time (temperature, chlorophyll, wind…). Keep a group only if it helps on 2024.
4. **Metrics:** macro-F1, balanced accuracy, recall on class 1 and class 2, and a confusion matrix per model. Also **onset recall**: weeks where `last_class = 0`, the only weeks where persistence can't help.
   - ⚠️ The regional **test year (2025) has only 1 class-1 week and 6 class-2 weeks**, so class-1 recall there is meaningless. Report class metrics on the site-week table (test: 1,378 / 49 / 28), or use `y_binary` for the regional table.
5. **Error priority:** missing a bloom (a false negative) is the costliest error and should be reported explicitly.
6. **Driver analysis:** compare logistic odds ratios, RF permutation importance, and boosting SHAP. Features that rank high across all three are the defensible top drivers.

## Beginner-Friendly Resources

- [An Introduction to Statistical Learning (free PDF, Python edition)](https://www.statlearning.com/): covers logistic regression, trees, random forests, boosting, and GAMs
- [StatQuest with Josh Starmer (YouTube)](https://www.youtube.com/@statquest): short visual explanations of each model
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
