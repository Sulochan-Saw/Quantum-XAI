| Date | Who | Exp ID | Command | Git commit | Seed | Output file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | Sukanta Mandal | DATA-01 | `python -m src.preprocessing.clean_data` | 3f891bb | 42 | `data/processed/*` | Trametinib, DepMap 25Q2, 712 lines, 498/107/107 |
| Date | Who | Exp ID | Command | Git commit | Seed | Output file | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | Sukanta Mandal | DATA-01 | `python -m src.preprocessing.clean_data` | 3f891bb | 42 | `data/processed/*` | Trametinib, DepMap 25Q2, 712 lines, 498/107/107 |
| 2026-11-16 05:02 | Member 4 | SANITY-SYN | `python -m src.quantum.sanity_check --seed 0` | n/a (not a git repo) | 0 | results/tables/sanity_synthetic.csv | synthetic data, debug only: val loss 0.768 -> 0.114 at epoch 30 (PASS); hybrid R2 0.885 vs LinearRegression 0.866 |
| 2026-11-16 05:03 | Member 4 | E2 | `python -m src.quantum.train --seed 0 --p_train 0.0 --name qnn_ideal` | n/a (not a git repo) | 0 | models/qnn_ideal_seed0.pt; results/tables/quantum_metrics.csv | best epoch 5, 30.0 s (2 CPUs), 41 params; test MAE 1.496, RMSE 2.045, R2 0.127 |
| 2026-11-16 05:04 | Member 4 | E2 | `python -m src.quantum.train --seed 1 --p_train 0.0 --name qnn_ideal` | n/a (not a git repo) | 1 | models/qnn_ideal_seed1.pt; results/tables/quantum_metrics.csv | best epoch 39, 29.8 s (2 CPUs), 41 params; test MAE 1.450, RMSE 2.015, R2 0.152 |
| 2026-11-16 05:04 | Member 4 | E2 | `python -m src.quantum.train --seed 2 --p_train 0.0 --name qnn_ideal` | n/a (not a git repo) | 2 | models/qnn_ideal_seed2.pt; results/tables/quantum_metrics.csv | best epoch 6, 33.7 s (2 CPUs), 41 params; test MAE 1.479, RMSE 2.036, R2 0.134 |
