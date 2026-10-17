# Forecasting harmful algal bloom risk near Sarasota

## Team Members

| Name | GitHub | Area |
|---|---|---|
| Anish KC | @mhosigiri | Data exploration, visualization, project coordination |
| Laura Williams | @laura-williams-1 | Data collection, exploratory analysis, documentation |
| Jerica Yuan | — | Data preprocessing, feature engineering, validation |
| Shreya R | — | Model development |
| Mohammed C | — | Model evaluation |

## Project Highlights

- Audited eight environmental data sources and compared three satellite chlorophyll products.
- Built two modelling tables from 2015 to May 2026: a **regional weekly table** (595 weeks, 486 labelled) and a **site-week table** (21,718 rows, 16,037 labelled, 136 sampling sites).
- Defined a next-week red tide risk label from NOAA *Karenia brevis* cell counts on the Anna Maria Island to Boca Grande coast, with 5,000 / 100,000 cells/L class cutoffs.
- Tested which features improve forecasts on unseen years: bloom history and duration plus season form the core set; environmental features are candidates to test.

## Project Overview

This Biointerphase project, part of Break Through Tech AI Studio, aims to forecast *K. brevis* (red tide) risk near Sarasota, Florida. The completed work covers data understanding, processing, and construction of the model table. 

## Data Exploration

We combined NOAA HABSOS cell counts with Sarasota Water Atlas measurements, Bradenton and KSRQ weather, offshore wind, Mote water temperature, and satellite chlorophyll. Cyanotoxin measurements are optional context; they do not define the red tide outcome. The CHIRPS precipitation file is not used: its dates are wrong after June 2016 (see notebook 02, section 8).

1. [Data understanding](notebooks/01_data_understanding.ipynb): checked coverage, missing values, units, outliers, and duplicates.
2. [Data processing](notebooks/02_data_processing.ipynb): cleaned each source and saved it as Parquet.
3. [Model tables](notebooks/03_model_table.ipynb): built the regional weekly and site-week tables, the next-week labels and the split.
4. [Exploratory data analysis](notebooks/04_exploratory_data_analysis.ipynb): EDA on the first (September) version of the weekly table.
5. [Feature engineering](notebooks/05_feature_engineering.ipynb): HAB history, duration, seasonal and temperature features, redundancy check, and the feature lists in `feature_sets.json`.

The classes are **0: below 5,000**, **1: 5,000–99,999**, and **2: at least 100,000 cells/L** (FWC shellfish-closure and satellite-visible bloom levels); `y_class_10k` keeps the earlier 10,000 cutoff. A week without a following-week sample has no label. The `split` column is set by the forecast week: train up to 2023, validate 2024, test 2025, holdout 2026. Column definitions are in `data/docs/data_dictionary.md`.

### Running the notebooks

Run 02, 03 and 05 in order. In Google Colab they read the raw files from `/content/drive/MyDrive/biointerphase_data/` (copy `data/water_atlas_stations.csv` there too). Locally they read [`data/`](data/) and write to `data/processed/` (not tracked by git). Install the packages with `pip install -r requirements.txt`.
