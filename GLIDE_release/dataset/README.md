# Dataset Directory

This release folder includes only the raw train/validation/test splits for the
four supported datasets: `Earthquake`, `COVID19`, `Citibike`, and `Crime`.

Expected layout for each dataset:

```text
dataset/
  Earthquake/
    data_train.pkl
    data_val.pkl
    data_test.pkl
```

The included files are:

- `data_train.pkl`, `data_val.pkl`, `data_test.pkl`: raw train/validation/test
  event sequences.

For 2D datasets, each raw event sequence should be a list of events in the form:

```text
[absolute_time, x_or_longitude, y_or_latitude]
```

Run preprocessing from the repository root:

```bash
python preprocess.py --dataset Earthquake --dim 2
python preprocess.py --dataset COVID19 --dim 2
python preprocess.py --dataset Citibike --dim 2
python preprocess.py --dataset Crime --dim 2
```

The current `preprocess.py` writes `data_*_processed1.pkl` and `stats1.pkl`.
Copy or rename them to `data_*_processed.pkl` and `stats.pkl` before running
`app.py`, unless you modify `data_loader()` to read the `*1.pkl` files.

Processed files, normalization statistics, and additional experimental variants
such as `processed1/2/3` or ablation-specific files are intentionally not
included here to keep the upload package clean.
