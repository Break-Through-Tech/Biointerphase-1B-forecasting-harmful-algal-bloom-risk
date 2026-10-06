# Project Status: Forecasting HAB Risk near Sarasota

**Program:** Break Through Tech AI Studio, Fall 2026 · **Partner:** Biointerphase
**Last updated:** 2026-10-06
**Current phase:** Moving from September (Data Understanding) to October (Model Development)

This document tracks progress, findings, blockers, and next steps. For the project brief see [Challenge-Project-Overview.md](Challenge-Project-Overview.md). For the full EDA write-up see [EDA_Findings_and_Recommendations.md](EDA_Findings_and_Recommendations.md).

---

## 1. Goal

Predict next week's *Karenia brevis* (red tide) risk class in the Sarasota study box (lat 26.5–28, lon −83.5 to −81.8) from current-week HAB, water quality, weather, and satellite data.

| Class | Meaning | Cells/L (next week's max) |
|---|---|---|
| 0 | No bloom | < 10,000 |
| 1 | At risk | 10,000–99,999 |
| 2 | Bloom | ≥ 100,000 |

Success is measured by classification performance (especially recall on the at-risk and bloom classes), interpretability of environmental drivers, and a reproducible Python workflow.

---

## 2. Milestone Status

| Month | Milestone | Status |
|---|---|---|
| September | Data Understanding | ✅ Done: data audit, cleaning, weekly model table, EDA |
| October | Model Development | ⏳ Starting: baselines and first models |
| November | Evaluation & Presentation | Not started |

---

## 3. Timeline of Work

| Date | Update | Who |
|---|---|---|
| Aug 20 | Repository created | Hari R. |
| Sep 15 | Team roster added to README | Anish KC |
| Sep 17 | Chlorophyll-a and additional wind data added | Alyssa L. (advisor) |
| Sep 25 | Chlorophyll NetCDF (7A) added; nutrients CSV normalized | Anish KC |
| Sep 29 | README updated with project highlights and pipeline | Anish KC |
| Oct 2 | CHIRPS precipitation data added | Alyssa L. (advisor) |
| Oct 3 | EDA notebook and findings summary merged (PR #19) | Mohammed C. |

---

## 4. What We Built in September

| Notebook | What it does |
|---|---|
| [01_data_understanding](notebooks/01_data_understanding.ipynb) | Audits all eight sources: coverage, units, outliers, duplicates, and the decision for each. Compares three satellite chlorophyll products. |
| [02_data_processing](notebooks/02_data_processing.ipynb) | Cleans each source with the decisions from 01 and saves it as Parquet. |
| [03_model_table](notebooks/03_model_table.ipynb) | Joins every source by week into the modeling table and creates the next-week label. |
| [04_exploratory_data_analysis](notebooks/04_exploratory_data_analysis.ipynb) | EDA on the final weekly table. |

**Output:** a weekly modeling table with **366 rows × 47 columns** (late 2018–2025).

**Time-based split:**

| Split | Years | Notes |
|---|---|---|
| Train | 2019–2021 | 141 labeled weeks |
| Excluded | 2022–2023 | HABSOS sampling too sparse |
| Validation | 2024 | |
| Test | 2025 | |

### Key data decisions

- **Label:** the highest HABSOS class seen the following week. Weeks with no sample the following week have no label, because no sample does not mean no bloom.
- **HABSOS:** keep Florida surface samples (≤ 2 m) inside the box. Recompute the category from cell count. For conflicting duplicates, keep the highest count.
- **Satellite chlorophyll:** keep **7A** (NOAA DINEOF gap-filled, 2 km). It has a value for almost every HABSOS sample, covers Sarasota Bay, and matches real observations from 7B and 7C (r ≈ 0.95). Drop 7B. Keep 7C only if 2010–2017 history is needed.
- **Chlorophyll anomaly:** the seasonal baseline uses 2018–2021 only, so it does not leak information from the test years.
- **Nutrients:** no station coordinates, so they are averaged at the region level. Values are carried forward up to 4 weeks. Below-detection values are set to MDL/2 and flagged.
- **Cyanotoxins:** freshwater toxins measured mostly in lakes. They are optional context and never part of the label.
- **Leakage:** every feature describes week *t* or earlier. The label is week *t+1*.

---

## 5. Key EDA Findings

1. **The target is imbalanced.** Training weeks are 64.5% Class 0, 6.4% Class 1 (only **9 weeks**), and 29.1% Class 2. Accuracy alone will be misleading.
2. **HAB conditions persist from week to week.** Class 0 stays Class 0 about 90% of the time, and Class 2 stays Class 2 about 93% of the time. A "same as this week" baseline reaches **86.0% train / 88.2% validation accuracy**. Any model has to beat this, especially in weeks where the class changes.
3. **Strongest environmental signals** (Spearman with next-week class):

   | Feature | ρ | Note |
   |---|---:|---|
   | Mote water temperature | −0.54 | Strong, but limited training coverage |
   | Air temperature | −0.29 to −0.31 | Cooler weeks come before higher classes |
   | Coastal chlorophyll anomaly | +0.29 | Above-seasonal chlorophyll comes before blooms |
   | 4-week rainfall | −0.19 | Weak and not monotonic |

4. **Blooms are seasonal.** Class 2 peaks October–January (60% of December weeks). There was no Class 2 in August or September during training. Part of the temperature signal may simply be seasonality.
5. **Some features have little or no training data.** KSRQ is 100% missing in training. Turbidity is almost empty. Cyanotoxins are patchy and missing in 2025. Mote is more than 50% missing. Several nutrients are 35–38% missing.

### Feature plan for the first model

- **Use:** current HAB class and cell count, coastal chlorophyll anomaly, Bradenton air temperature, rolling rainfall, offshore u-wind, and selected nutrients.
- **Sensitivity test only:** Mote water temperature.
- **Exclude:** KSRQ variables, turbidity, and cyanotoxins.
- **Try:** seasonal encodings (month, week-of-year sin/cos), compared with and without them.

---

## 6. Blockers and Open Questions

### Main blocker: handling missing data

Missing data is the biggest open problem before modeling. Coverage differs a lot between sources and years:

| Feature group | Missing in training (2019–2021) | Cause |
|---|---|---|
| KSRQ airport weather | 100% | Station data starts July 2022 |
| Turbidity (`nutr_turbidity_ntu`) | Almost 100% | Rarely measured |
| Mote water temperature | > 50% | Sensor starts Aug 2020; 364-day gap Mar 2023–Mar 2024; ends Mar 2025 |
| Several nutrients | ~35–38% | Sampled about monthly; carried forward up to 4 weeks |
| Cyanotoxins | High; none in 2025 | Data ends Jan 2025 |
| HABSOS (label) | 2022–2023 nearly empty | Sparse sampling; those years are excluded |

Filling a column that has no training data (like KSRQ) with an average does not help, because the model has no real values to learn from. The team still needs to decide on:

- **Which features to drop.** Current plan: KSRQ, turbidity and cyanotoxins.
- **How to fill the rest.** Options: median imputation, time-aware forward fill, missing-indicator columns, or models that handle NaN natively (e.g., HistGradientBoosting, XGBoost, LightGBM).
- **No leakage.** Imputation must be fitted on training data only, then applied to validation and test.

The choice should be tested, not assumed. Each modeler should report which approach they used (see Section 7).

### Other blockers and open questions

| # | Item | Impact | Owner / Action |
|---|---|---|---|
| 1 | **Class thresholds are provisional** (10k / 100k cells/L) | Defines the target | Confirm with advisor |
| 2 | **HABSOS 2022–2023 is nearly empty** | Two years lost from modeling | Ask advisor if a fuller extract exists |
| 3 | **Class 1 has only 9 training weeks** | The at-risk class may be very hard to learn | Try class weights or resampling, or consider merging classes |
| 4 | **Gridded wind has `uwnd` only, no `vwnd`** | No wind speed, direction, or alongshore wind | Ask the person who pulled the data to add `vwnd` and record the ERDDAP dataset ID |
| 5 | **Nutrient file has no station coordinates** | Nutrients can only be used at the region level | Request the Water Atlas station list from advisor |
| 6 | **Cyanotoxin data ends Jan 2025** even though the filename says 2026 | No test-period coverage | Ask advisor for a newer extract (low priority) |
| 7 | **CHIRPS precipitation (added Oct 2) is not in the pipeline yet** | Could replace or add to Bradenton rainfall | Add to notebooks 02 and 03 |
| 8 | **Reproducibility gaps:** notebooks need the Colab Drive path `/content/drive/MyDrive/biointerphase_data/`, `requirements.txt` is empty, and the processed Parquet and `docs/` outputs are not in the repo | Others cannot rerun the pipeline from the repo alone | Fill in `requirements.txt` and document how to set up the Drive folder |

---

## 7. Next Steps (October: Model Development)

**Plan:** each team member trains a different model, and we compare them to find the most effective one.

### Model assignments

| Member | Model | Notebook | Status |
|---|---|---|---|
| Anish KC | TBD | | Not started |
| Laura Williams | TBD | | Not started |
| Jerica Yuan | TBD | | Not started |
| Shreya R | TBD | | Not started |
| Mohammed C | TBD | | Not started |

Candidate models: logistic regression, random forest, gradient boosting (HistGradientBoosting / XGBoost / LightGBM), SVM, k-nearest neighbors, and a simple time-series or neural network model.

### Shared rules so the comparison is fair

Everyone uses the same:

- **Data:** the weekly model table from [03_model_table](notebooks/03_model_table.ipynb)
- **Split:** train 2019–2021, validate 2024, test 2025. Do not touch the test set until the final comparison
- **Starting features:** the feature plan in Section 5. Any additions or removals should be noted
- **Baselines to beat:** majority class and persistence ("same as this week": 88.2% validation accuracy)
- **Metrics:** macro F1, balanced accuracy, per-class precision and recall, a confusion matrix, and accuracy as a secondary metric. Also report results on **weeks where the class changes**

Each member should also report:

- How they handled missing data (see Section 6)
- How they handled the class imbalance (class weights, resampling, etc.)
- The most important features

### Other October tasks

- [ ] Agree on the missing-data approach (or compare a few across models)
- [ ] Assign one model per member
- [ ] Build a shared evaluation function so everyone reports metrics the same way
- [ ] Compare all models in one results table and pick the best for November
- [ ] Try with and without seasonal features and Mote temperature
- [ ] Integrate CHIRPS precipitation
- [ ] Resolve the advisor questions in Section 6

---

## 8. Team

| Name | Area |
|---|---|
| Anish KC | Data exploration, visualization, project coordination |
| Laura Williams | Data collection, exploratory analysis, documentation |
| Jerica Yuan | Data preprocessing, feature engineering, validation |
| Shreya R | Model development |
| Mohammed C | Model evaluation |

**Challenge Advisor:** Alyssa Long (Biointerphase) · **AI Studio Coach:** Eric Bayless
