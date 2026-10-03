# Exploratory Data Analysis (EDA) Findings and Recommendations

## Project Overview

This EDA examines the final weekly modeling table for the Sarasota harmful algal bloom (HAB) forecasting project.

The main forecasting question is:

> **Can current-week HAB and environmental conditions help predict the HAB class for the following week?**

Each row represents one week in the Sarasota study region. Environmental variables describe the current week, while `y_class` represents the maximum HAB class observed in the following week.

The main HAB classes are:

- **Class 0:** less than 10,000 cells/L
- **Class 1:** 10,000–99,999 cells/L
- **Class 2:** at least 100,000 cells/L

The time-based modeling split is:

- **Training:** 2019–2021
- **Excluded:** 2022–2023 because of low HABSOS coverage
- **Validation:** 2024
- **Test:** 2025

The final dataset contains **366 weekly rows and 47 columns**.

---

# 1. Target Class Distribution

## Finding

The training target is imbalanced.

Among the **141 labeled training weeks**:

| HAB Class | Weeks | Percent |
|---|---:|---:|
| Class 0 | 91 | 64.5% |
| Class 1 | 9 | 6.4% |
| Class 2 | 41 | 29.1% |

Class 1 is especially rare, with only **9 training examples**.

## Why this matters

A model could predict Class 0 most of the time and still appear to have reasonable accuracy because Class 0 is the most common class.

This means **accuracy alone should not be used to judge model performance**.

## Recommendation

During modeling, report:

- Macro F1-score
- Balanced accuracy
- Recall for each class
- Confusion matrix
- Overall accuracy as a secondary metric

Class 1 performance should be watched carefully because the model has very few examples from which to learn.

---

# 2. Current HAB Conditions Are Highly Persistent

## Finding

The current week's HAB class is strongly related to the following week's HAB class.

In the training data:

- Current **Class 0 stayed Class 0 about 89.8% of the time**
- Current **Class 2 stayed Class 2 about 92.5% of the time**
- Only **one training week** moved directly from current Class 0 to next-week Class 2

This shows that HAB conditions often persist from one week to the next.

## Why this matters

A very simple forecasting rule can already perform well:

> **Predict that next week's HAB class will be the same as this week's HAB class.**

This persistence baseline achieved approximately:

- **86.0% training accuracy**
- **88.2% validation accuracy**

A useful ML model should improve on these baselines, especially when the HAB class is changing.

---

# 3. Missing Data and Uneven Data Coverage

## Finding

Feature availability varies substantially across the study period.

Some of the main issues are:

- **KSRQ variables are 100% missing during training**
- `nutr_turbidity_ntu` is essentially unavailable during training
- Cyanotoxin variables have substantial missingness and are unavailable in the 2025 test period
- Mote water temperature is missing for more than half of the training period
- Several nutrient variables are approximately 35–38% missing during training

## Why this matters

A feature may look useful overall but still be unusable for modeling if it is not available during the training years.

For example, KSRQ begins after the training period. Therefore, the model does not have real KSRQ observations from which to learn a relationship with HAB class.

Simply filling these missing training values with an average would not solve the problem because the model would still have no real variation to learn from.

## Recommendation

### considering excluding

- KSRQ variables
- Turbidity
- Cyanotoxin variables
V

### Prefer features with more consistent coverage

- Bradenton weather variables
- Satellite chlorophyll variables
- Rolling rainfall
- Current HAB variables
- Offshore wind, while remembering only the east-west component is currently available

---

# 4. Feature Distributions

## Finding

The environmental variables have very different distributions.

Important examples:

### Current HAB cell counts

`hab_cellcount_max` is strongly right-skewed.

Most weeks have relatively lower cell counts, while a small number of weeks contain extremely high bloom concentrations.

### Rainfall

Rainfall variables are also right-skewed.

Many weeks have low or moderate rainfall, while fewer weeks contain very high rainfall amounts.

### Temperature

Air and water temperatures have a much narrower range and show clear seasonal behavior.

### Chlorophyll anomaly

`chl_coastal_anomaly` contains both negative and positive values.

- Negative = chlorophyll below the normal seasonal level
- Positive = chlorophyll above the normal seasonal level

## Recommendation

Extreme rainfall, chlorophyll, nutrients, or HAB cell counts may represent real environmental events rather than bad data.

---

# 5. Environmental Features vs. Next-Week HAB Class

Several current-week environmental variables showed differences across next-week HAB classes.

---

## Air Temperature

Median current-week air temperature was approximately:

| Next-Week Class | Median Air Temperature |
|---|---:|
| Class 0 | 26.7°C |
| Class 1 | 25.7°C |
| Class 2 | 22.2°C |

### Finding

Lower temperature tend to lead to higher HAB.

### Interpretation

Temperature may contain forecasting information, but part of this relationship may come from seasonality because Class 2 also occurred more often during cooler months.

---

## Mote Water Temperature

Mote water temperature showed one of the strongest environmental relationships with next-week HAB class.

However, it is only available for a limited part of the training period.

### Finding

Lower observed Mote water temperatures tended to occur before higher HAB classes.


---

## Coastal Chlorophyll Anomaly

Median coastal chlorophyll anomaly was approximately:

| Next-Week Class | Median Anomaly |
|---|---:|
| Class 0 | -0.32 mg/m³ |
| Class 1 | -0.25 mg/m³ |
| Class 2 | +0.13 mg/m³ |

### Finding

Postive coastal chlorophyll anamoly contain useful information for forecasting higher HAB conditions.

### Why this is interesting

The anomaly adjusts for normal seasonal chlorophyll variation.

Therefore, it may be more useful than raw chlorophyll concentration alone.

### Recommendation

Include `chl_coastal_anomaly` as one of the main candidate forecasting features.

---

## Four-Week Rainfall

Median 4-week rainfall was approximately:

| Next-Week Class | Median 4-Week Rainfall |
|---|---:|
| Class 0 | 73.9 mm |
| Class 1 | 173.0 mm |
| Class 2 | 51.3 mm |

### Finding

Rainfall differs across classes, but there is no simple increasing or decreasing pattern.

Class 1 has the highest median rainfall, but there are only nine Class-1 training examples.

### Recommendation

Keep rolling rainfall features available for modeling, but do not expect rainfall alone to clearly separate HAB severity.

---

## Corrected Chlorophyll-a from Nutrient Data

Median `nutr_chla_corr_ugL` was approximately:

| Next-Week Class | Median Corrected Chlorophyll-a |
|---|---:|
| Class 0 | 16.1 µg/L |
| Class 1 | 11.6 µg/L |
| Class 2 | 24.9 µg/L |

### Finding

Class-2 weeks were preceded by higher median corrected chlorophyll-a.

### Caution


Nutrient measurements have missing values, and recent measurements may be carried forward for up to four weeks.

---

# 6. Spearman Correlation Results

Spearman correlation was used because many features are skewed and `y_class` is ordered from 0 to 2.

The strongest environmental relationships included:

| Feature | Approx. Spearman Correlation with `y_class` |
|---|---:|
| Mote water temperature | -0.54 |
| Air temperature variables | -0.29 to -0.31 |
| Coastal chlorophyll anomaly | +0.29 |
| 4-week rainfall | -0.19 |

## Interpretation

Mote water temperature showed the strongest environmental association with next-week HAB class (Spearman ≈ −0.54), with lower observed water temperatures tending to occur before higher HAB classes.

Air temperature showed a moderate negative association with next-week HAB class (approximately −0.29 to −0.31). Higher HAB classes tended to follow cooler weeks, supporting the pattern observed in the boxplots.

Coastal chlorophyll anomaly showed a positive association with next-week HAB class (Spearman ≈ +0.29). Weeks with chlorophyll levels above their seasonal baseline tended to be followed by higher HAB classes, suggesting that chlorophyll anomaly may provide useful forecasting information.

Four-week rainfall showed negative association with next-week HAB class (approximately −0.19). This mean lower accumulated rainfall before higher classes.

## Recommendation

Prioritize further modeling experiments with:

- Current HAB class
- Current HAB cell count
- Coastal chlorophyll anomaly
- Air temperature
- Rolling rainfall
- Selected nutrient variables
- Mote water temperature as an optional/sensitivity feature

---

# 7. Seasonality

## Finding

Next-week Class 2 occurred more often in late fall and winter during the training years.

The highest observed Class-2 percentages were approximately:

| Month | Class-2 Share |
|---|---:|
| December | 60.0% |
| November | 44.4% |
| January | 41.7% |
| October | 40.0% |

No Class-2 target weeks occurred in August or September in the 2019–2021 training sample.

## Why this matters

Temperature also follows a strong seasonal cycle.

Therefore, the relationship between lower temperature and higher HAB class may partly reflect the fact that both temperature and HAB severity vary with time of year.

## Recommendation

Consider including seasonal information such as:

- Month
- Week of year
- Sine/cosine seasonal encoding

However, compare models with and without explicit seasonal features to determine whether they improve forecasting.

---

# 8. Most Important EDA Findings

The EDA suggests the following main conclusions:

1. **The target is imbalanced**, especially Class 1.
2. **Current HAB class is a very strong predictor of next week's class.**
3. **A persistence baseline is difficult to beat and must be included.**
4. **Several features have major missing-data or coverage problems.**
5. **KSRQ variables cannot be learned normally with the current training period.**
6. **Temperature and coastal chlorophyll anomaly show some of the clearest environmental relationships with next-week HAB class.**
7. **The Mote temperature signal looks strong but is based on limited data.**
8. **Rainfall and individual nutrient measurements show weaker or more complicated relationships.**
9. **HAB severity and many environmental variables are seasonal.**

---

# 9. Features to Avoid in the First Model

Based on the EDA, the first modeling attempt should probably avoid:

- KSRQ variables because there is no training coverage
- `nutr_turbidity_ntu` because it is almost entirely missing
- Cyanotoxin variables because of poor and inconsistent coverage


---

# 10. Missing-Data Strategy

For usable variables that still contain missing values:

1. Measure missingness in the training set.
2. Fit all imputation rules using **training data only**.
3. Apply the learned imputation values to validation and test.
4. Do not calculate imputation statistics from the full dataset because that would leak future information into training.

Possible strategies include:

- Median imputation
- Simple time-aware forward filling where scientifically justified
- Missing-indicator features for selected variables
- Models that can natively handle missing values

The final method should be evaluated rather than assumed to be best.

---

# 11. Evaluation Strategy

Because of class imbalance, report more than accuracy.

Recommended metrics:

### Overall

- Accuracy
- Balanced accuracy
- Macro F1-score

### Per class

- Precision
- Recall
- F1-score

### Visualization

- Confusion matrix

Pay special attention to:

- Class 1 recall
- Class 2 recall
- weeks where the HAB class changes


