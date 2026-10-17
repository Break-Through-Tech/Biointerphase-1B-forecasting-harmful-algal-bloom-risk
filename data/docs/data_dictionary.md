# Data dictionary

Built by `notebooks/03_model_table.ipynb`; `05_feature_engineering.ipynb` adds the engineered features and `feature_sets.json`.

## model_table_weekly.parquet

One row per week (Monday start), 2015-01-05 to 2026-05-25. Features describe week t, labels week t+1. Label area: the focus coast (26.70-27.55 N, east of 82.95 W).

| column                        | unit          | source         | description                                                                                                  |
|:------------------------------|:--------------|:---------------|:-------------------------------------------------------------------------------------------------------------|
| week_start                    | date          | -              | Monday of predictor week t (local date, America/New_York)                                                    |
| target_week                   | date          | -              | Monday of the forecast week t+1                                                                              |
| target_year                   | year          | -              | Year of target_week; assign test years with this, not week_start                                             |
| bloom_season                  | year          | -              | Bloom season of target_week: Aug 1 of Y to Jul 31 of Y+1 is season Y                                         |
| split                         | -             | -              | By target_year: train (<= 2023), validate (2024), test (2025), holdout (2026)                                |
| y_class                       | class         | HABSOS         | LABEL: max class in the focus coast in week t+1 (0 < 5k, 1 = 5k-99,999, 2 >= 100k cells/L); NaN if unsampled |
| y_binary                      | 0/1           | HABSOS         | LABEL: y_class >= 1 (at risk or bloom)                                                                       |
| y_log_cells                   | log10 cells/L | HABSOS         | log10(max cells/L + 1) in week t+1                                                                           |
| y_n_samples                   | count         | HABSOS         | Surface samples behind the label (0 if unsampled)                                                            |
| y_n_sites                     | count         | HABSOS         | Distinct sites behind the label                                                                              |
| label_effort_ok               | 0/1           | HABSOS         | 1 if y_n_samples >= 3 (suggested reliability filter)                                                         |
| y_class_10k                   | class         | HABSOS         | Alternative label with the earlier cutoffs (0 < 10k, 1 = 10k-99,999, 2 >= 100k)                              |
| y_class_box                   | class         | HABSOS         | Alternative label: whole 26.5-28 N box                                                                       |
| y_class_mote_25km             | class         | HABSOS         | Alternative label: samples within 25 km of Mote Marine Laboratory                                            |
| hab_class                     | class         | HABSOS         | Max class observed in week t (NaN if unsampled; use last_class as the feature)                               |
| n_habsos_samples              | count         | HABSOS         | Focus-coast samples in week t                                                                                |
| n_habsos_sites                | count         | HABSOS         | Distinct sites sampled in week t                                                                             |
| n_habsos_days                 | count         | HABSOS         | Distinct sampling days in week t                                                                             |
| hab_pos_share                 | share         | HABSOS         | Share of week-t samples >= 5k cells/L                                                                        |
| hab_bloom_share               | share         | HABSOS         | Share of week-t samples >= 100k cells/L                                                                      |
| hab_detect_share              | share         | HABSOS         | Share of week-t samples with any K. brevis (> 0 cells/L)                                                     |
| hab_log_cells_p90             | log10 cells/L | HABSOS         | log10 of the 90th-percentile cell count in week t                                                            |
| last_class                    | class         | HABSOS         | Last observed weekly class up to and including week t (carried forward)                                      |
| last_log_cells                | log10 cells/L | HABSOS         | log10(max cells/L + 1) of the last observed week                                                             |
| weeks_since_obs               | weeks         | HABSOS         | Weeks since the last sampled week (0 = sampled this week)                                                    |
| class_max_4wk                 | class         | HABSOS         | Max observed class over weeks t-3..t                                                                         |
| log_cells_trend               | log10 cells/L | HABSOS         | last_log_cells minus its value two weeks earlier                                                             |
| nutr_age_weeks                | weeks         | Water Atlas    | Weeks since the newest nutrient anomaly of any group                                                         |
| rain_total_mm                 | mm            | Bradenton COOP | Rain in week t (NaN if < 5 days reported; record ends 2025-12-31)                                            |
| rain_days                     | days          | Bradenton COOP | Days in week t with > 0.25 mm rain                                                                           |
| rain_2wk_mm                   | mm            | Bradenton COOP | Rain over weeks t-1..t                                                                                       |
| rain_4wk_mm                   | mm            | Bradenton COOP | Rain over weeks t-3..t                                                                                       |
| rain_8wk_mm                   | mm            | Bradenton COOP | Rain over weeks t-7..t                                                                                       |
| air_tmax_c                    | deg C         | Bradenton COOP | Highest daily max air temperature in week t                                                                  |
| air_tmin_c                    | deg C         | Bradenton COOP | Lowest daily min air temperature in week t                                                                   |
| air_tmean_c                   | deg C         | Bradenton COOP | Mean of daily (max + min) / 2 in week t                                                                      |
| air_trange_c                  | deg C         | Bradenton COOP | Mean daily range (max - min) in week t                                                                       |
| ksrq_pressure_hpa             | hPa           | KSRQ           | Mean sea-level pressure, weekly mean (reference only, starts Jul 2022)                                       |
| ksrq_air_temp_c               | deg C         | KSRQ           | Air temperature, weekly mean (reference only)                                                                |
| ksrq_dew_point_c              | deg C         | KSRQ           | Dew point, weekly mean (reference only)                                                                      |
| ksrq_wind_speed               | m/s           | KSRQ           | Wind speed, weekly mean (reference only)                                                                     |
| ksrq_wind_u                   | m/s           | KSRQ           | East-west wind component, weekly mean (reference only)                                                       |
| ksrq_wind_v                   | m/s           | KSRQ           | North-south wind component, weekly mean (reference only)                                                     |
| has_ksrq                      | 0/1           | KSRQ           | 1 if KSRQ has data in week t                                                                                 |
| uwnd_box                      | m/s           | Gridded wind   | East-west wind over the 7 x 7 grid, weekly mean (+ = westerly = onshore)                                     |
| uwnd_offshore                 | m/s           | Gridded wind   | East-west wind over the three western grid columns, weekly mean                                              |
| uwnd_onshore_days             | days          | Gridded wind   | Days in week t with uwnd_box > 0                                                                             |
| uwnd_westerly_mean            | m/s           | Gridded wind   | Weekly mean of max(uwnd_box, 0): onshore part only                                                           |
| uwnd_easterly_mean            | m/s           | Gridded wind   | Weekly mean of max(-uwnd_box, 0): offshore part only                                                         |
| uwnd_box_4wk                  | m/s           | Gridded wind   | Mean of uwnd_box over weeks t-3..t (>= 2 weeks)                                                              |
| mote_temp_mean_c              | deg C         | Mote           | Water temperature at 2 m, weekly mean (reference only; use air_tmean_4wk)                                    |
| mote_temp_max_c               | deg C         | Mote           | Water temperature at 2 m, weekly max (reference only)                                                        |
| has_mote                      | 0/1           | Mote           | 1 if Mote has data in week t                                                                                 |
| chl_box_med                   | mg/m3         | Chlorophyll 7A | Weekly median of the daily median over all Gulf-connected ocean pixels                                       |
| chl_coast20_med               | mg/m3         | Chlorophyll 7A | Same, ocean pixels within 20 km of the coast                                                                 |
| chl_focus_0_10_med            | mg/m3         | Chlorophyll 7A | Same, focus-latitude pixels 0-10 km from the coast                                                           |
| chl_focus_10_40_med           | mg/m3         | Chlorophyll 7A | Same, focus-latitude pixels 10-40 km from the coast                                                          |
| chl_focus_0_40_med            | mg/m3         | Chlorophyll 7A | Same, focus-latitude pixels 0-40 km from the coast                                                           |
| chl_box_n_valid               | pixels        | Chlorophyll 7A | Weekly mean of valid ocean pixels per day                                                                    |
| chl_coast20_anom              | log10         | Chlorophyll 7A | Mean log10 anomaly vs the 2018-2021 day-of-year climatology, within 20 km                                    |
| chl_focus_0_10_anom           | log10         | Chlorophyll 7A | Same, focus 0-10 km                                                                                          |
| chl_focus_0_40_anom           | log10         | Chlorophyll 7A | Same, focus 0-40 km                                                                                          |
| chl_focus_frac_gt3            | share         | Chlorophyll 7A | Share of focus 0-40 km pixels with chl > 3 mg/m3                                                             |
| chl_focus_frac_gt10           | share         | Chlorophyll 7A | Share of focus 0-40 km pixels with chl > 10 mg/m3                                                            |
| chl_qc_box_cover              | share         | Chlorophyll 7A | Weekly mean share of valid ocean pixels (days < 0.8 are NaN in the zone columns)                             |
| chl_focus_0_40_anom_4wk       | log10         | Chlorophyll 7A | Mean of chl_focus_0_40_anom over weeks t-3..t (>= 2 weeks)                                                   |
| chl_focus_0_10_med_4wk        | mg/m3         | Chlorophyll 7A | Mean of chl_focus_0_10_med over weeks t-3..t                                                                 |
| chl_focus_10_40_med_4wk       | mg/m3         | Chlorophyll 7A | Mean of chl_focus_10_40_med over weeks t-3..t                                                                |
| chl_focus_0_40_med_chg        | log10         | Chlorophyll 7A | Week-on-week change of log10 chl_focus_0_40_med                                                              |
| nutr_bay_chla_corr_ugL_anom   | log10         | Water Atlas    | bay stations: mean station-month anomaly of chla_corr_ugL (vs 2006-2021), carried <= 5 weeks                 |
| nutr_bay_cond_umho_anom       | log10         | Water Atlas    | bay stations: mean station-month anomaly of cond_umho (vs 2006-2021), carried <= 5 weeks                     |
| nutr_bay_do_mgL_anom          | log10         | Water Atlas    | bay stations: mean station-month anomaly of do_mgL (vs 2006-2021), carried <= 5 weeks                        |
| nutr_bay_nox_mgL_anom         | log10         | Water Atlas    | bay stations: mean station-month anomaly of nox_mgL (vs 2006-2021), carried <= 5 weeks                       |
| nutr_bay_po4_mgL_anom         | log10         | Water Atlas    | bay stations: mean station-month anomaly of po4_mgL (vs 2006-2021), carried <= 5 weeks                       |
| nutr_bay_tkn_mgL_anom         | log10         | Water Atlas    | bay stations: mean station-month anomaly of tkn_mgL (vs 2006-2021), carried <= 5 weeks                       |
| nutr_bay_tn_mgL_anom          | log10         | Water Atlas    | bay stations: mean station-month anomaly of tn_mgL (vs 2006-2021), carried <= 5 weeks                        |
| nutr_bay_tp_mgL_anom          | log10         | Water Atlas    | bay stations: mean station-month anomaly of tp_mgL (vs 2006-2021), carried <= 5 weeks                        |
| nutr_bay_water_temp_c_anom    | deg C         | Water Atlas    | bay stations: mean station-month anomaly of water_temp_c (vs 2006-2021), carried <= 5 weeks                  |
| nutr_fresh_chla_corr_ugL_anom | log10         | Water Atlas    | fresh stations: mean station-month anomaly of chla_corr_ugL (vs 2006-2021), carried <= 5 weeks               |
| nutr_fresh_cond_umho_anom     | log10         | Water Atlas    | fresh stations: mean station-month anomaly of cond_umho (vs 2006-2021), carried <= 5 weeks                   |
| nutr_fresh_do_mgL_anom        | log10         | Water Atlas    | fresh stations: mean station-month anomaly of do_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_fresh_nh3_mgL_anom       | log10         | Water Atlas    | fresh stations: mean station-month anomaly of nh3_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_fresh_nox_mgL_anom       | log10         | Water Atlas    | fresh stations: mean station-month anomaly of nox_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_fresh_po4_mgL_anom       | log10         | Water Atlas    | fresh stations: mean station-month anomaly of po4_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_fresh_tkn_mgL_anom       | log10         | Water Atlas    | fresh stations: mean station-month anomaly of tkn_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_fresh_tn_mgL_anom        | log10         | Water Atlas    | fresh stations: mean station-month anomaly of tn_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_fresh_tp_mgL_anom        | log10         | Water Atlas    | fresh stations: mean station-month anomaly of tp_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_fresh_water_temp_c_anom  | deg C         | Water Atlas    | fresh stations: mean station-month anomaly of water_temp_c (vs 2006-2021), carried <= 5 weeks                |
| nutr_tidal_chla_corr_ugL_anom | log10         | Water Atlas    | tidal stations: mean station-month anomaly of chla_corr_ugL (vs 2006-2021), carried <= 5 weeks               |
| nutr_tidal_cond_umho_anom     | log10         | Water Atlas    | tidal stations: mean station-month anomaly of cond_umho (vs 2006-2021), carried <= 5 weeks                   |
| nutr_tidal_do_mgL_anom        | log10         | Water Atlas    | tidal stations: mean station-month anomaly of do_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_tidal_nh3_mgL_anom       | log10         | Water Atlas    | tidal stations: mean station-month anomaly of nh3_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_tidal_nox_mgL_anom       | log10         | Water Atlas    | tidal stations: mean station-month anomaly of nox_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_tidal_po4_mgL_anom       | log10         | Water Atlas    | tidal stations: mean station-month anomaly of po4_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_tidal_tkn_mgL_anom       | log10         | Water Atlas    | tidal stations: mean station-month anomaly of tkn_mgL (vs 2006-2021), carried <= 5 weeks                     |
| nutr_tidal_tn_mgL_anom        | log10         | Water Atlas    | tidal stations: mean station-month anomaly of tn_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_tidal_tp_mgL_anom        | log10         | Water Atlas    | tidal stations: mean station-month anomaly of tp_mgL (vs 2006-2021), carried <= 5 weeks                      |
| nutr_tidal_water_temp_c_anom  | deg C         | Water Atlas    | tidal stations: mean station-month anomaly of water_temp_c (vs 2006-2021), carried <= 5 weeks                |
| nutr_bay_n_stations           | count         | Water Atlas    | bay stations sampled in week t                                                                               |
| nutr_fresh_n_stations         | count         | Water Atlas    | fresh stations sampled in week t                                                                             |
| nutr_tidal_n_stations         | count         | Water Atlas    | tidal stations sampled in week t                                                                             |

## site_week_table.parquet

One row per (site_key, week_start) with a current sample or a next-week label. The regional columns (wind, rain, air temperature, chlorophyll) are copied from the same week of the regional table.

| column                | unit          | source         | description                                                          |
|:----------------------|:--------------|:---------------|:---------------------------------------------------------------------|
| site_key              | -             | HABSOS         | Site id: latitude_longitude rounded to 0.01 deg (~1 km)              |
| site_class_now        | class         | HABSOS         | Site's max class in week t (NaN if unsampled)                        |
| site_log_cells_now    | log10 cells/L | HABSOS         | Site's log10(max cells/L + 1) in week t                              |
| site_last_class       | class         | HABSOS         | Site's last observed class (carried forward)                         |
| site_last_log_cells   | log10 cells/L | HABSOS         | Site's last observed log10 count                                     |
| site_weeks_since_obs  | weeks         | HABSOS         | Weeks since the site was last sampled                                |
| nbr_class             | class         | HABSOS         | Max class this week at other sites within 20 km (carried <= 2 weeks) |
| reg_last_class        | class         | HABSOS         | Regional last_class of the same week                                 |
| reg_class_max_4wk     | class         | HABSOS         | Regional class_max_4wk of the same week                              |
| reg_n_samples         | count         | HABSOS         | Regional n_habsos_samples of the same week                           |
| km_to_pixel           | km            | Chlorophyll 7A | Distance from the site to its nearest Gulf pixel                     |
| pixel_dist_coast_km   | km            | Chlorophyll 7A | Distance of that pixel from the coast                                |
| inshore               | bool          | Chlorophyll 7A | Site is > 2.5 km from any Gulf pixel (bay or pass)                   |
| latitude              | deg           | HABSOS         | Mean site latitude                                                   |
| longitude             | deg           | HABSOS         | Mean site longitude                                                  |
| site_log_chl          | log10 mg/m3   | Chlorophyll 7A | Weekly mean log10 chl, 3 x 3 pixels around the nearest Gulf pixel    |
| site_log_chl_anom     | log10         | Chlorophyll 7A | Same for the day-of-year anomaly                                     |
| site_log_chl_chg      | log10         | Chlorophyll 7A | Week-on-week change of site_log_chl                                  |
| site_log_chl_anom_4wk | log10         | Chlorophyll 7A | Mean of site_log_chl_anom over weeks t-3..t                          |
| y_class               | class         | HABSOS         | LABEL: the site's max class in week t+1                              |
| y_class_10k           | class         | HABSOS         | Same with the 10k / 100k cutoffs                                     |