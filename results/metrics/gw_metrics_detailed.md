# GW metrics – 2022/23

Hold-out season 2022/23: best GW model by MAE is gbm_seasonal (MAE=1.122, RMSE=2.025, R2=0.263, Spearman=0.693, Bias=0.036, MedAE=0.462, P90_AE=2.772).

| model_key           |   mae |   rmse |    r2 |   spearman |   bias |   medae |   p90_ae |   rank_mae |
|:--------------------|------:|-------:|------:|-----------:|-------:|--------:|---------:|-----------:|
| gbm_seasonal        | 1.122 |  2.025 | 0.263 |      0.693 |  0.036 |   0.462 |    2.772 |          1 |
| linear_seasonal     | 1.217 |  2.083 | 0.22  |      0.666 |  0.045 |   0.46  |    2.918 |          2 |
| linear_anytime_lag5 | 1.219 |  2.086 | 0.22  |      0.666 |  0.045 |   0.46  |    2.923 |          3 |
| linear_anytime_lag3 | 1.248 |  2.105 | 0.204 |      0.67  |  0.057 |   0.566 |    2.938 |          4 |
