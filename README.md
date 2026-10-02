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
- Cleaned the sources and built a weekly Sarasota modeling table: **366 rows, 47 columns**.
- Defined a next-week red tide risk label from NOAA *Karenia brevis* cell counts. **244 weeks** have labels in the planned training, validation, and test splits.

## Project Overview

This Biointerphase project, part of Break Through Tech AI Studio, aims to forecast *K. brevis* (red tide) risk near Sarasota, Florida. The completed work covers data understanding, processing, and construction of the model table. 

## Data Exploration

We combined NOAA HABSOS cell counts with Sarasota Water Atlas measurements, Bradenton and KSRQ weather, offshore wind, Mote water temperature, and satellite chlorophyll. Cyanotoxin measurements are optional context; they do not define the red tide outcome.

1. [Data understanding](notebooks/01_data_understanding.ipynb): checked coverage, missing values, units, outliers, and duplicates.
2. [Data processing](notebooks/02_data_processing.ipynb): cleaned each source and saved it as Parquet.
3. [Model table](notebooks/03_model_table.ipynb): joined the sources by week and set the label to the highest HABSOS class observed the following week.

The provisional classes are **0: below 10,000**, **1: 10,000–99,999**, and **2: at least 100,000 cells/L**. A week without a following-week sample has no label. HABSOS sampling is sparse in 2022–2023, so those weeks are excluded from the planned model splits.

The notebooks run in Google Colab. They read raw files from `/content/drive/MyDrive/biointerphase_data/` and save cleaned files and the weekly table in its `processed/` folder. Raw inputs are also in [`data/`](data/).
