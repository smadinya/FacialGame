# Datasets

Neither dataset is committed. The scripts expect this layout, with all paths set in
`src/config.py`:

```
data/
├── train/{happy,neutral,surprise}/     5,000 FER2013 images (1,666 / 1,666 / 1,668)
├── test/{happy,neutral,surprise}/      3,838 FER2013 images (1,774 / 1,233 / 831)
└── transfer/
    ├── train/{cat,dog}/                1,000 each
    └── test/{cat,dog}/                 200 each
```

## Facial expressions (FER2013)

1. Download [FER2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and extract
   it to `data/kaggle/`, so that `data/kaggle/train/happy/` and `data/kaggle/test/happy/` exist.
2. Run:

   ```sh
   python src/export_dataset.py
   ```

   This copies a balanced 5,000-image training subset of the three classes into `data/train/`
   and the complete test split for those classes into `data/test/`.

The images are 48×48 grayscale. `src/preprocess.py` loads them at 150×150 RGB.

The test split is imbalanced: `happy` makes up 46.2% of it, so a model that always answers
`happy` scores 46%. Read the test accuracies against that floor, not against 33%.

## Dog vs Cat (transfer task)

1. Download Microsoft's
   [Cats and Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765), the
   Asirra set that Kaggle's *Dogs vs Cats* is built from. It needs no Kaggle login. Extract it so
   that `data/PetImages/Cat/` and `data/PetImages/Dog/` exist.
2. Run:

   ```sh
   python src/prepare_transfer.py            # or pass the PetImages path as an argument
   ```

   Some files in the archive are truncated or aren't really JPEGs, and they crash training partway
   through an epoch. The script decodes every candidate image first and skips the bad ones (17 in
   our copy). It then copies the first 1,200 usable images per class: 1,000 for training and 200
   for test.
