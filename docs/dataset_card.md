# Dataset Card: Trametinib

- **Drug:** Trametinib (MEK inhibitor), chosen for its many cell lines and wide LN_IC50 spread
- **Response data:** GDSC2 fitted dose response (GDSC2_fitted_dose_response*.xlsx); target = LN_IC50
- **Expression data:** DepMap Public 25Q2, OmicsExpressionProteinCodingGenesTPMLogp1.csv (TPM from Salmon v1.10.0, log1p), rows keyed by ACH- ModelID
- **Matching:** DepMap ModelID mapped to SangerModelID through Model.csv (DepMap Public 25Q2), joined to GDSC2 SANGER_MODEL_ID
- **Cell lines before matching:** 966 with Trametinib response in GDSC2; 1,673 in the DepMap expression file
- **After matching:** 712 cell lines, 19,193 genes (254 GDSC2 Trametinib lines had no matching DepMap expression profile)
- **Exclusions:** rows with missing LN_IC50 or missing Sanger ID, duplicate cell lines, genes with missing values
- **Split:** random 70/15/15, seed 42 -> 498 train / 107 val / 107 test
- **Feature selection (train only):** variance prefilter (top 2000 genes), F-score, correlation filter < 0.9
- **Scaling:** MinMax fit on train only, clipped to [0, 1]
- **8 genes (in order):** ETV4, DUSP6, SPRY2, TYR, LYZ, CAPN3, S100B, ETV5
- **Leakage controls:** no AUC, RMSE or Z_SCORE features; all fitting on train only
- **Limitations:** TYR, S100B and LYZ are lineage markers; simulator-based research, not clinical