# ML RI Prediction

This repository contains machine-learning workflows for tropical cyclone rapid intensification (RI) prediction using SHIPS/HWRF-style environmental predictors. The project includes training datasets, individual storm case files, pre-trained Keras models, exploratory notebooks, and scripts used to evaluate Hurricane Otis (2023), Hurricane Patricia (2015), and 2023 real-time forecast cases.

## Description

The central research question is how environmental predictors at the RI onset time compare with predictors along the future 24-hour tropical cyclone track for RI classification. The current workflow treats RI as a binary classification problem and trains or evaluates three neural-network-style classifiers:

- dense logistic-style neural network;
- simple recurrent neural network (RNN);
- gated recurrent unit (GRU).

The code uses SHIPS diagnostic predictors from HWRF/HAFS-style forecast guidance. The main 24-hour dataset includes predictors at 00, 06, 12, 18, and 24 h lead times. The Otis and Patricia case files are used to compare a failed RI forecast case against a better-predicted RI case and to test sensitivity to SST, shear, and storm translation speed.

## How To Run

Create a Python environment. TensorFlow is the heaviest dependency, so a conda environment is usually the most reliable option on HPC systems.

```bash
conda create -n ri-prediction python=3.10
conda activate ri-prediction
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyter ipython
python -m pip install tensorflow requests wget
python -m pip install -e .
```

A pip-only installation can start with:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run from the repository root.

Train the main 24-hour models:

```bash
python scripts/train_ri_models.py
```

Evaluate the Hurricane Otis case with the pre-trained models:

```bash
python scripts/predict_otis.py
```

Run the combined training and Otis sensitivity workflow used by the job scripts:

```bash
python scripts/train_and_predict_otis.py
```

Open notebooks:

```bash
jupyter lab notebooks/
```

Fetch SHIPS diagnostics for a storm lifetime example:

```bash
python scripts/fetch_ship.py lifetime 13 PATRICIA20E 2015102106
```

On IU HPC systems, job templates are in `jobs/`. They now change into the repository root relative to the job-script location and call the moved scripts.

## Directory Structure

```text
RI-prediction/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── archive/
│   └── backup/
├── data/
│   ├── README.md
│   ├── cases/
│   ├── examples/
│   └── training/
├── docs/
│   └── references/
│       ├── AMS-36-Hurricane Confrence-poster-LowryKieu-Apr18.pdf
│       └── README.md
├── jobs/
│   ├── RI_getSHIP.sh
│   ├── job_br200.sh
│   └── job_carbonate.sh
├── models/
│   ├── README.md
│   └── pretrained/
├── notebooks/
│   ├── README.md
│   └── *.ipynb
├── results/
│   └── README.md
├── scripts/
│   ├── README.md
│   ├── fetch_ship.py
│   ├── predict_otis.py
│   ├── train_and_predict_otis.py
│   ├── train_ri_models.py
│   └── legacy/
└── src/
    └── ri_prediction/
        ├── __init__.py
        ├── models.py
        └── utils.py
```

## Data And Models

The main training file used by the current scripts is:

- `data/training/SHIP_allbasin_2011_2022_Version4.csv`

The main case-study files are:

- `data/cases/OTIS18E_master.csv`
- `data/cases/PATRICIA20E_master.csv`

Pre-trained Keras files are stored in `models/pretrained/` for both 00 h and 24 h experiments:

- `RI_model_logistics_00h.keras`, `RI_model_RNN_00h.keras`, `RI_model_GRU_00h.keras`
- `RI_model_logistics_24h.keras`, `RI_model_RNN_24h.keras`, `RI_model_GRU_24h.keras`

## References

Primary project reference:

Lowry, X., and C. Kieu, 2024: Extreme Rapid Intensification of Hurricanes Otis (2023) and Patricia (2015): A Machine Learning Diagnosis. AMS 36th Hurricane and Tropical Meteorology Conference, Long Beach, CA. Local copy: `docs/references/AMS-36-Hurricane Confrence-poster-LowryKieu-Apr18.pdf`.

Related references listed in the poster:

1. Du, T. D., T. Ngo-Duc, M. T. Hoang, and C. Q. Kieu, 2013: A study of connection between tropical cyclone track and intensity errors in the WRF model. Meteorology and Atmospheric Physics, 122, 55-64. https://doi.org/10.1007/s00703-013-0278-0
2. Fan, W.-T., et al., 2021: Hitting time of rapid intensification onset in hurricane-like vortices. Physics of Fluids. https://doi.org/10.1063/5.0062119
3. Kieu, C., et al., 2021: On the track-dependence of the tropical cyclone intensity forecast errors in the COAMPS-TC model. Weather and Forecasting. https://doi.org/10.1175/WAF-D-20-0085.1
4. Kieu, C., et al., 2013: Vertical structure of tropical cyclones at onset of the rapid intensification in the HWRF model. Geophysical Research Letters, 9, 3298-3306. https://doi.org/10.1002/2014GL059584
5. Patra, M., W.-T. Fan, and C. Kieu, 2022: Sensitivity of tropical cyclone intensity variability to different stochastic parameterization methods. Frontiers in Earth Science, 10. https://doi.org/10.3389/feart.2022.893781

## Contact

Chanh Kieu  
Department of Earth and Atmospheric Sciences, Indiana University  
Email: ckieu@indiana.edu
