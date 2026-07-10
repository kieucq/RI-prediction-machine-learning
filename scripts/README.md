# Scripts

Primary scripts:

- `train_ri_models.py`: train logistic, RNN, and GRU RI classifiers from the main SHIPS training dataset.
- `predict_otis.py`: load pretrained models and evaluate Hurricane Otis case predictions.
- `train_and_predict_otis.py`: combined training and Otis prediction workflow used by the job scripts.
- `fetch_ship.py`: download and convert NCEP/HWRF/HAFS SHIPS diagnostic files.

`legacy/` contains earlier exploratory scripts and notebook exports retained for provenance.
