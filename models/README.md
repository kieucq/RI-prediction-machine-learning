# Models

- `pretrained/`: Keras models checked into this project for 00 h and 24 h RI prediction experiments.
- `checkpoints/` and `generated/`: recommended local destinations for models created during retraining. These directories are ignored by git.

The primary scripts read and write model files through `models/pretrained/` by default so old top-level model paths are no longer required.
