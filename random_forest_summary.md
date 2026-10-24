# Random Forest Model: Progress Summary

**Project:** Forecasting Harmful Algal Bloom (HAB) Risk, Biointerphase × Break Through Tech AI Studio
**Notebook:** `notebooks/random_forest.ipynb`
**Branch:** `random-forest-model`
**Status:** First model trained and compared with the baseline. Tuning and driver analysis are next.

---

## 1. Goal

Predict the _K. brevis_ bloom class at each site **one week ahead**:

| Class | Meaning  | Cell count (cells/L) |
| ----- | -------- | -------------------- |
| 0     | No bloom | under 5,000          |
| 1     | At risk  | 5,000 – 99,999       |
| 2     | Bloom    | 100,000 or more      |

The label is `y_class`: the class at the same site in the following week. Every feature only uses information available at the current week, so the model never sees the future.

---

## 2. Setup (local, VS Code)

Teammates using Colab don't need this; Colab already has these packages.

- Installed `pyarrow` (needed to read `.parquet` files) and `scikit-learn` into the project's `.venv`.
- Versions: pyarrow 26.0.0, scikit-learn 1.9.1.
- scikit-learn 1.4 or newer lets the random forest handle missing values directly, so **no filling in of missing values was needed**.
- Note: `scikit-learn` is **not** listed in `requirements.txt` yet. It should be added for local users.
- Note for Colab: the notebook loads data from `../data/final/...`. Colab users need to change this path to wherever the data lives in their session or Drive.

---

## 3. Data used

**Table:** `data/final/site_week_table_fe.parquet`. Each row is one **site-week** (one sampling location in one week).

We used this table rather than the regional one because it has far more rows, so the class-level scores are more reliable.

| Number         | Meaning                                                                       |
| -------------- | ----------------------------------------------------------------------------- |
| 21,718 rows    | All site-weeks                                                                |
| 46 columns     | Labels, features, IDs, split                                                  |
| 16,037 labeled | Rows where the next week was sampled. The rest were dropped: no answer to learn from |

### Class counts by time split

The data is split **by year**, not randomly, so the model is always tested on years it hasn't seen.

| Split    | Years  | No bloom (0) | At risk (1) | Bloom (2) | Total  |
| -------- | ------ | ------------ | ----------- | --------- | ------ |
| train    | ≤ 2023 | 9,783 (79%)  | 1,229 (10%) | 1,367 (11%) | 12,379 |
| validate | 2024   | 1,485 (90%)  | 123 (7%)    | 45 (3%)   | 1,653  |
| test     | 2025   | 1,378 (95%)  | 49 (3%)     | 28 (2%)   | 1,455  |
| holdout  | 2026   | 550 (100%)   | 0           | 0         | 550    |

**Key points:**

- **The classes are imbalanced.** Most weeks have no bloom. A model that always says "no bloom" would be 95% accurate on 2025 and useless, so we judge models on **recall for classes 1 and 2** and **macro-F1**, not accuracy.
- **Blooms become rarer in later years** (21% of training rows, but 5% of 2025). The model has to cope with this shift.
- **2025 has only 49 at-risk and 28 bloom rows**, so a few predictions can swing the scores.
- **2026 has no blooms**, so it can only measure false alarms.

---

## 4. Features (`site_core`, 12 features)

All 12 describe **bloom history**: at this site, nearby and recently, plus the time of year.

| Group                  | Feature                | Meaning                                                                     |
| ---------------------- | ---------------------- | --------------------------------------------------------------------------- |
| This site's history    | `site_last_class`      | Class at the site's most recent sample                                      |
|                        | `site_last_log_cells`  | Cell count at that sample (log10 scale)                                     |
|                        | `site_weeks_since_obs` | Weeks since the site was last sampled                                       |
| Neighbors and region   | `nbr_class`            | Highest class this week at other sites within 20 km (reused for up to 2 weeks) |
|                        | `reg_last_class`       | Most recent class for the whole focus coast                                 |
|                        | `reg_class_max_4wk`    | Highest regional class over the last 4 weeks                                |
| Bloom streaks (regional) | `class_run_weeks`    | Weeks in a row the current class has lasted (max 52)                        |
|                        | `weeks_since_bloom`    | Weeks since the last bloom week of 100,000 cells/L or more (max 104)        |
|                        | `pos_weeks_12wk`       | How many of the last 12 weeks reached 5,000 cells/L or more                 |
| Context                | `reg_n_samples`        | Number of water samples taken along the coast that week                     |
|                        | `season_sin`, `season_cos` | Time of year as a point on a circle (keeps Dec 31 next to Jan 1)        |

**Important:**

- **None of these are environmental causes** (temperature, wind, chlorophyll, nutrients). Those are in `site_candidates` and will be added one group at a time to test which ones improve predictions. This is how we identify the **top drivers**.
- **`reg_n_samples` needs caution.** Agencies sample more when they suspect a bloom, so this may reflect people reacting to a bloom rather than the environment causing one.

### Missing values in the core features

| Feature                                      | Missing | Likely reason                                        |
| -------------------------------------------- | ------- | ---------------------------------------------------- |
| `weeks_since_bloom`                          | 1,111   | Start of the record, before the first bloom in the data |
| `nbr_class`                                  | 268     | No neighboring site was sampled                      |
| `site_last_class`, `site_last_log_cells`, `site_weeks_since_obs` | 111 each | The site's first sample, so there's no earlier one |

These values are missing for a reason, not at random. The random forest handles them directly.

---

## 5. Missing data check (merged dataset)

A teammate's earlier list of missing-data issues was checked against the final tables:

| Issue                              | Status in the final data                                           |
| ---------------------------------- | ------------------------------------------------------------------ |
| KSRQ weather 100% missing in training | Excluded (`do_not_use`); not in the site-week table             |
| `nutr_turbidity_ntu` unavailable   | Dropped                                                            |
| Cyanotoxins missing in 2025        | Dropped                                                            |
| Mote water temperature missing     | Excluded (`do_not_use`); air temperature (`air_tmean_4wk`) used instead |
| Nutrients 35–38% missing           | Reduced in the regional tables; **not in the site-week table at all** |

**Gaps still remaining** (matter when environmental features are added later):

| Feature group      | Missing in training | Missing in 2024 and 2025 |
| ------------------ | ------------------- | ------------------------ |
| Wind (`uwnd_*`)    | 60%                 | 0%                       |
| Chlorophyll        | 43%                 | 0%                       |
| Rain and air temperature | ~0%           | 0% (but 90–100% missing in 2026) |

**Why this matters:** these features are missing because their data sources start in later years (for example, satellite chlorophyll starts in 2018), not at random. A model could learn that "missing chlorophyll means older years", which says nothing about the ocean. **When testing wind or chlorophyll, train only on the years where that data exists.**

**Impact on the current model:** none. The 12 core features are almost complete.

---

## 6. Models and results (validation year 2024)

### Baseline: "next week = last week"

Predict that each site's class next week will be the same as its last observed class. Nothing is learned. This is the minimum score any model has to beat.

### Random forest

| Setting            | Value        | Why                                           |
| ------------------ | ------------ | --------------------------------------------- |
| `n_estimators`     | 500          | 500 decision trees vote on each prediction    |
| `class_weight`     | `"balanced"` | Counts rare bloom weeks more heavily          |
| `min_samples_leaf` | 10           | Each tree rule must cover at least 10 rows, which prevents memorizing |
| `random_state`     | 42           | Same results every run                        |

Trained on years up to 2023 using the 12 core features, and scored on 2024.

### Comparison

| Measure                                  | Baseline  | Random forest | Better            |
| ---------------------------------------- | --------- | ------------- | ----------------- |
| Macro-F1                                 | **0.490** | 0.472         | Baseline, slightly |
| Balanced accuracy                        | 0.478     | **0.499**     | Forest, slightly  |
| Recall, at risk (1)                      | 0.301     | **0.341**     | Forest            |
| Recall, bloom (2)                        | 0.178     | **0.244**     | Forest            |
| **Onset recall** (new blooms, out of 90) | 0.000     | **0.511**     | **Forest, by a lot** |
| Blooms missed completely (true 2, predicted 0) | 19  | **14**        | Forest            |
| False alarms (true 0, predicted 1 or 2)  | **64**    | 133           | Baseline          |

**Confusion matrices.** Rows show what really happened; columns show what was predicted.

Baseline:

|        | pred 0 | pred 1 | pred 2 |
| ------ | ------ | ------ | ------ |
| true 0 | 1421   | 51     | 13     |
| true 1 | 71     | 37     | 15     |
| true 2 | 19     | 18     | 8      |

Random forest:

|        | pred 0 | pred 1 | pred 2 |
| ------ | ------ | ------ | ------ |
| true 0 | 1352   | 45     | 88     |
| true 1 | 30     | 42     | 51     |
| true 2 | 14     | 20     | 11     |

### What the results mean

- **The forest catches new blooms.** In 2024 there were 90 cases where a site had no bloom last time but did the next week. The baseline caught none (copying last week can't predict something new); the forest caught about half (46). This is the main value of the model.
- **It raises more false alarms.** It predicted bloom or risk in 133 weeks that turned out fine, about twice as many as the baseline.
- **It struggles to tell "at risk" from "bloom".** Of 45 real bloom weeks, it labeled only 11 as bloom and 20 as at risk. It usually notices something is happening but often gets the severity wrong.
- **Overall scores are similar** because the extra false alarms cancel out the extra catches in macro-F1.

### Summary sentence

> The random forest catches about half of new bloom onsets, which the persistence baseline cannot catch at all, and misses fewer blooms. The trade-off is roughly twice as many false alarms, so overall scores are similar. Next, we tune the model and test environmental drivers to improve on this.

---

## 7. Next steps

1. **Permutation importance:** find which of the 12 features the forest relies on most (feeds the "top drivers" section).
2. **Tuning:** adjust `min_samples_leaf`, `max_depth` and `max_features` on 2024 only, to try to cut false alarms without losing new-bloom catches.
3. **Test without `site_last_class`:** measure how much of the skill is just persistence.
4. **Add environmental features one group at a time** (temperature, chlorophyll, wind, rain), training only on years where each has data. Keep a group only if it improves 2024 scores.
5. **Final score on 2025 (test)** once, with the chosen model.
6. Open question from the team plan: are some sites only sampled when a bloom is suspected? (Compare sample counts in bloom and non-bloom weeks.)

---

## 8. Notebook cell guide

| Cell | What it does                                                        |
| ---- | ------------------------------------------------------------------- |
| 1    | Loads the site-week table and `feature_sets.json`                   |
| 2    | Keeps only rows with a label (`y_class`)                            |
| 3    | Counts each class in each split                                     |
| 4    | Lists the 12 core features and their missing values                 |
| 5    | Checks the scikit-learn version                                     |
| 6    | Defines `evaluate()`, the scoring function used for every model     |
| 7    | Splits the data into train (≤ 2023) and validate (2024)             |
| 8    | Scores the baseline ("next week = last week")                       |
| 9    | Trains the random forest and scores it on 2024                      |
