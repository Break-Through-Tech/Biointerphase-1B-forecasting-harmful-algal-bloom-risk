# Random Forest Model: Progress Summary

**Project:** Forecasting Harmful Algal Bloom (HAB) Risk, Biointerphase × Break Through Tech AI Studio
**Notebook:** `notebooks/random_forest.ipynb`
**Branch:** `random-forest-model`
**Status:** Model adjusted to the client's feedback (section 8): early-warning metrics with a 3:1 miss cost, threshold tuning, rolling validation, a regional model, environmental feature tests, a site reliability check and a one-time 2025 test.

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

## 7. What the forest relies on (permutation importance)

### How it works

Permutation importance asks: **"If I scramble one feature, how much worse does the model get?"**

1. Take one feature and shuffle its values randomly between rows, so it no longer matches the right week.
2. Ask the trained forest to predict 2024 again. The model is **not** retrained.
3. If the score (macro-F1) drops a lot, the forest was relying on that feature. If it barely changes, the forest wasn't using it.
4. Repeat 10 times for each of the 12 features and average.

We used this instead of the built-in `feature_importances_`, which tends to rank features with many distinct values (like cell counts) too high.

**How to read the numbers:**

- **Positive:** scrambling it hurt the score, so the forest relies on it.
- **Near zero:** scrambling it made no difference, so the forest isn't really using it.
- **Negative:** scrambling it _improved_ the score. The forest learned something from this feature in the older years that doesn't hold in 2024.
- **Spread:** how much the 10 shuffles disagreed. If the spread is as big as the importance, the result isn't reliable.

### Results (2024)

| Group               | Feature                | Importance | Spread | Reliable?                     |
| ------------------- | ---------------------- | ---------- | ------ | ----------------------------- |
| ✅ Helps            | `pos_weeks_12wk`       | +0.033     | 0.005  | Yes, the strongest            |
|                     | `season_cos`           | +0.026     | 0.009  | Yes                           |
|                     | `site_last_class`      | +0.009     | 0.002  | Yes, small                    |
| ➖ No clear effect  | `site_last_log_cells`  | +0.006     | 0.008  | No, spread is bigger          |
|                     | `reg_class_max_4wk`    | +0.001     | 0.003  | No, about zero                |
|                     | `class_run_weeks`      | −0.004     | 0.006  | No, too noisy                 |
|                     | `nbr_class`            | −0.005     | 0.009  | No, too noisy                 |
|                     | `reg_n_samples`        | −0.009     | 0.009  | No, too noisy                 |
|                     | `site_weeks_since_obs` | −0.010     | 0.008  | No, too noisy                 |
|                     | `season_sin`           | −0.017     | 0.016  | No, too noisy                 |
| ❌ Hurts            | `reg_last_class`       | −0.010     | 0.005  | Probably, borderline          |
|                     | `weeks_since_bloom`    | −0.025     | 0.007  | Yes, clearly hurts            |

### What this means in plain English

**The three clues the forest actually uses:**

1. **`pos_weeks_12wk`, "How busy have the last 3 months been?"** It counts how many of the last 12 weeks had at least 5,000 cells/L along the coast. This is the most useful clue: if the coast has been active recently, a bloom is more likely next week.
2. **`season_cos`, "What time of year is it?"** This half of the season encoding mainly separates winter from summer. The forest has learned that bloom risk depends on the season.
3. **`site_last_class`, "What did this site look like last time?"** This is the "copy last week" information, and it helps only a little. The forest leans more on regional activity and season than on simply copying the site's last reading.

**The clue that hurts:**

- **`weeks_since_bloom`** makes the forest worse on 2024. The forest learned a pattern from it in the older years that doesn't hold in 2024. Likely reasons:
  - the training years had many more blooms than 2024, so "time since the last bloom" behaved differently
  - about 9% of its training values are missing (start of the record), but none are missing in 2024, so the forest may have learned something about those blanks that doesn't apply later
- **`reg_last_class`** shows the same problem, more weakly.

**The rest** (7 features) make no clear difference. Their results are too noisy to say whether they help or hurt.

### Cautions

1. **This shows how the model predicts, not what causes blooms.** All 12 features are bloom history and season. The finding is "recent regional bloom activity and time of year are the best predictors", **not** "these cause blooms". Causes need the environmental features (temperature, wind, chlorophyll), which come later.
2. **Similar features share credit.** For example, `site_last_class` and `site_last_log_cells` carry almost the same information. When one is scrambled, the forest can use the other, so both can look less important than they really are.
3. **Don't over-tune on 2024.** If we drop features because they hurt on 2024, 2024 will then look better than it should. The fair check is 2025, used only once at the end.

### Summary sentence

> Permutation importance shows the forest relies mainly on recent regional bloom activity (`pos_weeks_12wk`), time of year (`season_cos`) and, to a lesser extent, the site's last observed class. `weeks_since_bloom` reduced performance on 2024, which suggests patterns from earlier years don't carry over to recent years.

---

## 8. Part 2: Adjusting to the client's feedback

### What the client asked for

1. **Onset recall as a primary metric**, reported alongside bloom misses and false alarms
2. **A regional next-week forecast as the main product**; site-level stays secondary until it is shown to be reliable across sites
3. **Misses cost more than false alarms**, starting at 2:1 or 3:1, with the raw counts still shown
4. **Environmental features only if they help validation**, with what was tested documented

### What changed

| Before (Part 1) | Now (Part 2) |
| --- | --- |
| Judged on macro-F1 | Judged on **onset recall** and a **cost score**: 3 × misses + false alarms (2:1 also reported) |
| Forecast = predicted class | Forecast = **alert** ("next week will be at risk or bloom, 5,000+ cells/L") when the forest's probability of class 1 or 2 is above a **threshold** |
| Validated on 2024 only | **Rolling validation**: each of 2019, 2020, 2021 and 2024 is predicted by a forest trained only on earlier years, then the results are added together |
| Site-level only | **Regional model** (main) and site model (secondary) |

**Definitions** (the `early_warning_report()` function):

- **Accuracy, balanced accuracy, macro-F1:** the same three-class scores as Part 1. When the forest raises an alert, it predicts "bloom" (2) if it thinks class 2 is more likely than class 1, and "at risk" (1) otherwise. They're shown for completeness, but because no-bloom weeks dominate, accuracy mostly rewards staying quiet, so models are **not** chosen by them.
- **Onset:** the site (or region) had no bloom last time, but next week is at risk or bloom
- **Miss:** a week that was really at risk or bloom, with no alert
- **Bloom miss:** a real bloom week (class 2) with no alert, the worst kind of miss
- **False alarm:** an alert in a week that turned out fine

**Why 3:1 is the main ratio:** at 2:1, the cheapest regional threshold catches almost no onsets, which works against the client's early-warning priority. Both costs are shown everywhere.

**Why 2022 and 2023 aren't validation years:** they have almost no labels (15 and 48 site-weeks, 8 and 17 regional weeks). They're still used for training.

**How the threshold is chosen:** the threshold with the lowest 3:1 cost wins. If other thresholds are within 5% of that cost, the one that catches the most onsets is used. This puts onset recall first without ignoring cost.

**How much the results move by chance:** the same forest was rerun with 4 random seeds. A change only counts as real if it moves the lowest cost by more than **twice** the seed-to-seed range: **50** at site level, **6** regionally.

### Rescoring Part 1 on 2024 (site level)

| Model | Accuracy | Balanced acc. | Macro-F1 | Onset recall | Onsets caught | Misses | Bloom misses | False alarms | Cost 2:1 | Cost 3:1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (next week = last week) | **0.887** | 0.478 | **0.490** | 0.00 | 0 / 90 | 90 | 19 | **64** | 244 | 334 |
| Random forest (Part 1) | 0.850 | **0.499** | 0.472 | **0.51** | **46 / 90** | **44** | **14** | 133 | **221** | **265** |

The baseline has higher accuracy because it raises fewer alarms. Once misses cost more, the forest is clearly better.

### Site-level model (rolling validation, 2019–2021 + 2024)

| Model | Threshold | Accuracy | Balanced acc. | Macro-F1 | Onset recall | Onsets caught | Misses | Bloom misses | False alarms | Cost 2:1 | Cost 3:1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | - | 0.896 | 0.524 | 0.518 | 0.00 | 0 / 291 | 291 | 74 | 302 | 884 | 1175 |
| Forest, balanced, leaf 10 | 0.50 | 0.871 | 0.621 | 0.530 | 0.55 | 161 / 291 | 138 | 31 | 546 | 822 | 960 |
| **Forest, balanced, leaf 30 (chosen)** | **0.60** | 0.874 | 0.619 | 0.532 | 0.52 | **151 / 291** | 151 | 33 | 520 | 822 | 973 |
| Forest, no weights, leaf 10 | 0.20 | 0.859 | 0.602 | 0.495 | 0.51 | 149 / 291 | 150 | 33 | 587 | 887 | 1037 |
| Forest, no weights, leaf 30 | 0.20 | 0.863 | 0.600 | 0.495 | 0.52 | 152 / 291 | 146 | 32 | 561 | 853 | 999 |

- The forest catches **about half of all onsets** (the baseline catches none) and **cuts total misses roughly in half** (151 vs 291).
- Over the four years, the forest also beats the baseline on **balanced accuracy** (0.62 vs 0.52) and **macro-F1** (0.53 vs 0.52). Its plain accuracy is lower (0.87 vs 0.90) because it raises more alarms.
- The four forest versions are within noise of each other. Leaf 30 was kept because larger leaves are less likely to memorize.
- **The threshold is the main dial.** The notebook prints the full table, for example: 0.40 catches 72% of onsets with 855 false alarms, and 0.70 catches 43% with 382. This is the table to show the client so they can choose where to sit.

### Regional model (main product)

One row per week for the whole focus coast. Across the four validation years there are only **14 regional onsets**, so a difference of one or two weeks isn't meaningful.

| Model | Threshold | Accuracy | Balanced acc. | Macro-F1 | Onset recall | Onsets caught | Misses | Bloom misses | False alarms | Cost 2:1 | Cost 3:1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | - | **0.792** | 0.694 | **0.695** | 0.00 | 0 / 14 | 14 | 2 | **13** | **41** | 55 |
| **Regional forest** | 0.40 | 0.732 | **0.705** | 0.678 | **0.43** | **6 / 14** | **8** | 1 | 29 | 45 | **53** |
| Site forest, combined (highest site probability each week) | 0.25 | 0.721 | 0.686 | 0.673 | 0.36 | 5 / 14 | 9 | 0 | 29 | 47 | 56 |

**What this means:**

- **At the regional level, the forest does not clearly beat "next week = last week" on cost.** The costs are within the noise range (±6). The overall scores are also close: the baseline is ahead on accuracy and macro-F1, the forest slightly ahead on balanced accuracy.
- **What the forest adds is early warning.** It catches 6 of 14 onsets, which the baseline can't catch at all, at the price of about 16 more false alarms over four years (about 4 per year).
- Combining the site forest into a regional forecast is no better than the regional forest.
- The likely cause is the small regional dataset: 364 training weeks, compared with 12,000+ site-weeks.

### Feature tests (documented as the client asked)

Each test is compared with the core-feature forest **trained on the same years**. Chlorophyll only exists from 2018 and wind from 2019, so for those, both forests train only on years with data. Otherwise the test measures "less training data", not the feature. The numbers are the lowest 3:1 cost (negative change = better).

**Site level** (a change has to beat 50 to count):

| Test | Trained from | Core cost | Test cost | Change | Onset recall (core → test) | Kept? |
| --- | --- | --- | --- | --- | --- | --- |
| Drop `weeks_since_bloom` | 2015 | 934 | 910 | −24 | 0.52 → 0.47 | No (within noise) |
| Drop `reg_last_class` | 2015 | 934 | 903 | −31 | 0.52 → 0.55 | No (within noise) |
| Drop `site_last_class` | 2015 | 934 | 930 | −4 | 0.52 → 0.59 | No |
| Add temperature | 2015 | 934 | 923 | −11 | 0.52 → 0.57 | No |
| Add chlorophyll | 2018 | 1002 | 996 | −6 | 0.40 → 0.42 | No |
| Add wind | 2019 | 667 | 664 | −3 | 0.59 → 0.58 | No |
| Add rain | 2015 | 934 | 941 | +7 | 0.52 → 0.54 | No |

**Regional** (a change has to beat 6 to count):

| Test | Trained from | Core cost | Test cost | Change | Kept? |
| --- | --- | --- | --- | --- | --- |
| Drop `weeks_since_bloom` | 2015 | 53 | 51 | −2 | No |
| Drop `last_class` | 2015 | 53 | 53 | 0 | No |
| Add temperature | 2015 | 53 | 50 | −3 | No |
| Add chlorophyll | 2018 | 68 | 77 | +9 | No (hurts) |
| Add wind | 2019 | 50 | 47 | −3 | No |
| Add bay nutrients | 2015 | 53 | 52 | −1 | No |
| Add rain | 2015 | 53 | 52 | −1 | No |

**Conclusion:** no environmental group and no drop improved validation by more than chance, so **both final models use only the core bloom-history and season features**. Following the client's guidance, environmental variables are not forced in.

Two side findings:

- Dropping `site_last_class` barely changes the site cost: the forest gets the same "last week" information from `site_last_log_cells`.
- Permutation importance (below) flagged `season_cos`, `pos_weeks_12wk`, `reg_n_samples` and `class_run_weeks`. Retraining without each one, or all four together, lowered cost by only 25–41, which is still within noise, so they stay.

### What the final models rely on (cost-based permutation importance)

Same idea as section 7, but the score is the **lowest 3:1 cost over all four validation years**, with the threshold re-picked after each scramble. Positive = the cost goes up when the feature is scrambled = the forest relies on it.

**Site model:**

| Feature | Cost increase | Spread | Reading |
| --- | --- | --- | --- |
| `nbr_class` | +184 | 21 | **By far the most important:** the highest class at other sites within 20 km this week |
| `site_last_log_cells` | +38 | 14 | The site's last cell count |
| `weeks_since_bloom` | +15 | 12 | Small; spread is nearly as large |
| All others | −28 to +3 | | No clear help |

**This is different from section 7**, where `pos_weeks_12wk` and `season_cos` led. Section 7 scored macro-F1 on 2024 only. Scored on early-warning cost over four years, the forest's onset skill comes mainly from **"is a nearby site already showing cells?"** In practice, that means blooms spread along the coast and neighbors give the warning.

**Regional model:** every feature is within ±4 cost. With 14 onsets, no single feature stands out.

### Site-level reliability across the coast

Site rolling-validation results split into thirds by site latitude (every site is marked offshore, so an inshore/offshore split isn't possible):

| Part of coast | Sites | Accuracy | Balanced acc. | Macro-F1 | Onset recall | Onsets caught | Misses | Bloom misses | False alarms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| South | 46 | 0.872 | 0.613 | 0.566 | 0.49 | 62 / 127 | 73 | 25 | 176 |
| Central | 45 | 0.860 | 0.579 | 0.493 | 0.58 | 54 / 93 | 39 | 8 | 182 |
| North | 45 | 0.888 | 0.665 | 0.498 | 0.49 | 35 / 71 | 39 | 0 | 162 |

- Onset recall is **similar along the coast** (0.49–0.58).
- **Most bloom misses are in the south** (25 of 33).
- Only **21 of 136 sites** have 5 or more onsets. At those, onset recall ranges from **0.17 to 0.80** (median 0.40). **Reliability at individual sites hasn't been shown yet**, so site-level stays secondary, as the client proposed.

### Final check on 2025 (test year, used once)

The settings, features and thresholds were all fixed by the rules above before this step. Each model was retrained on all years up to 2024.

| Model | Accuracy | Balanced acc. | Macro-F1 | Onset recall | Onsets caught | Misses | Bloom misses | False alarms | Cost 2:1 | Cost 3:1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Site: baseline | **0.907** | 0.499 | **0.463** | 0.00 | 0 / 33 | 33 | 11 | **81** | **147** | 180 |
| Site: forest (t = 0.60) | 0.869 | **0.592** | 0.444 | **0.85** | **28 / 33** | **5** | **3** | 156 | 166 | **171** |
| Regional: baseline | **0.962** | **0.659** | **0.637** | n/a | 0 / 0 | 0 | 0 | **1** | **1** | **1** |
| Regional: forest (t = 0.40) | 0.904 | 0.637 | 0.626 | n/a | 0 / 0 | 0 | 0 | 4 | 4 | 4 |
| Regional: site forest combined (t = 0.25) | 0.827 | 0.463 | 0.514 | n/a | 0 / 0 | 0 | 0 | 5 | 5 | 5 |

- **Site level:** the forest caught **28 of 33 new blooms** and cut misses from 33 to 5, with about twice the false alarms. It wins at 3:1 and loses at 2:1.
- **Regional:** 2025 had **no regional onsets**. Its 7 risky weeks all followed risky weeks, so "next week = last week" was nearly perfect, and **2025 can't measure regional early warning**. The forest's cost is 3 extra false alarms over the year.
- Note: the 2025 cell was first run by mistake before the feature tests were finished. Nothing was changed because of it, and the feature rules ended up keeping the core features anyway, so the final numbers are the same.

### Summary sentence

> Scored on early warning with misses costing 3× a false alarm, the site-level random forest catches about half of new bloom onsets in rolling validation (and 28 of 33 in 2025), where the persistence baseline catches none. Its main signal is bloom activity at neighboring sites. At the regional level, the forest is roughly tied with persistence on cost but catches 6 of 14 onsets the baseline misses, for about 4 extra false alarms a year. No environmental feature group improved validation by more than chance, so the final models use bloom-history and season features only.

---

## 9. Next steps

1. **Choose the operating point with the client.** Show the threshold trade-off tables (site and regional) and confirm that 3:1 is the right cost ratio.
2. **Get more regional signal.** 14 regional onsets is too few to separate models. Options: a different regional label (`y_class_box`, `y_class_mote_25km`), or a regional forecast built from neighbor activity, since `nbr_class` carries the site model.
3. **Site reliability:** look at the south, where most bloom misses happen, and the sites with low onset recall.
4. **2026 holdout:** it has no blooms, so it can only count false alarms. Air temperature and rain are mostly missing in 2026, which doesn't affect the core-feature models.
5. Open question from the team plan: are some sites only sampled when a bloom is suspected? (`reg_n_samples` / `n_habsos_samples` reflect sampling effort.)

---

## 10. Notebook cell guide

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
| 10   | Permutation importance: which features the forest relies on         |

**Part 2** (after the original cells):

| Section | What it does |
| ------- | ------------ |
| Early-warning report card | `early_warning_report()`: accuracy, balanced accuracy, macro-F1, onset recall, misses, bloom misses, false alarms, cost 2:1 and 3:1. `to_class()` turns probabilities into a class (0/1/2). Rescores the Part 1 baseline and forest |
| Rolling validation | `rolling_predictions()`, `threshold_sweep()`, `pick_threshold()`, `best_cost()` |
| Site-level model | Compares forest settings, checks seed noise, prints the threshold trade-off table |
| Feature tests (site) | Drops and environmental groups, then the final site feature set |
| Site reliability | Results by part of the coast and per site |
| Regional model | Baseline vs regional forest vs combined site forest, with trade-off tables |
| Feature tests (regional) | Same tests for the regional model |
| Permutation importance | Cost-based, for the final site and regional models |
| Final check on 2025 | One-time test of the chosen models |
