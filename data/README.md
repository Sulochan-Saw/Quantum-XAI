# Data folder

Dataset for the Quantum-XAI project: predicting ln(IC50) of **Trametinib** from gene expression.
Full details (counts, selection method, limitations) are in `docs/dataset_card.md`.

## Sources (raw data, not committed to Git)

Place these in `data/raw/`:

| File | Source |
| --- | --- |
| `GDSC2_fitted_dose_response*.xlsx` | GDSC2 fitted dose response (Cancer RxGene), release: <fill in> |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | DepMap Public 25Q2 (TPM, log1p) |
| `Model.csv` | DepMap Public 25Q2, maps ModelID (ACH-) to SangerModelID |

Raw files are large and have licence terms, so they are listed in `.gitignore`.

## Processed data (committed): `data/processed/`

| File | Content |
| --- | --- |
| `X_train.npy`, `X_val.npy`, `X_test.npy` | Features, shape (n, 8), values in [0, 1] |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | Target: raw LN_IC50, shape (n,) |
| `feature_names.json` | The 8 gene names, in column order |
| `meta.json` | Drug, split sizes, seed, number of genes, train mean and std of y |
| `scaler.joblib` | MinMax scaler, fitted on train only |
| `test_cell_lines.csv` | SANGER_MODEL_ID and CELL_LINE_NAME of the test rows |

Split: 498 train / 107 validation / 107 test (random 70/15/15, seed 42, 712 cell lines).

Columns are ordered by feature importance, best first:
ETV4, DUSP6, SPRY2, TYR, LYZ, CAPN3, S100B, ETV5.
The 4-qubit model uses the first 4 columns, the 8-qubit model uses all 8.

## How to load

```python
from src.data_io import load_data
Xtr, Xva, Xte, ytr, yva, yte = load_data()      # 8 features
Xtr, Xva, Xte, ytr, yva, yte = load_data(k=4)   # first 4 features
```

## How to regenerate

From the repo root, with the virtual environment active:

```
python -m src.preprocessing.clean_data
python -m pytest tests/test_preprocessing.py
```

The drug and seed are set in `config.yaml` (`drug_name`, `seed`). Output is identical for the same raw files and seed.

## Rules

- Target is LN_IC50 only. No AUC, RMSE, Z_SCORE or concentration columns are used as features.
- Feature selection and scaling are fitted on the training rows only.
- Test rows are never used for any decision.