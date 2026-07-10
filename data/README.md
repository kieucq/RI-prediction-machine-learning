# Data

This directory contains SHIPS/HWRF-derived predictor tables used by the rapid-intensification workflows.

- `training/`: multi-storm training datasets. `SHIP_allbasin_2011_2022_Version4.csv` is the main 00-24 h dataset used by the current scripts.
- `cases/`: individual storm case files, including Hurricane Otis (2023) and Hurricane Patricia (2015), plus 2023 real-time cases.
- `examples/`: small example input files for 00 h, 12 h, and 24 h predictor formats.

Most CSV files include a final `class` column where `1` denotes rapid intensification and `0` denotes non-RI. Missing values in the original SHIPS products are represented with `?` and are converted to `-99999` by `ri_prediction.utils.filterdata`.
