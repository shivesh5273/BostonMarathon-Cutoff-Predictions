# Which models best predict acceptance?

We reframed the problem as **Accepted vs Not Accepted**, using per-year/per-age-bin/per-gender thresholds.

Train/test: trained on earlier years, tested on the latest year we have rows for.


## TL;DR

- **Best overall ROC-AUC:** **DecisionTree** (AUC 1.000, Acc 1.000, F1 1.000, Brier 0.000)


## Full metrics table

| model                    |   accuracy |       f1 |   roc_auc |       brier |
|:-------------------------|-----------:|---------:|----------:|------------:|
| DecisionTree             |   1        | 1        |  1        | 0           |
| RandomForest             |   1        | 1        |  1        | 0.000101002 |
| RandomForest(calibrated) |   1        | 1        |  1        | 4.06033e-07 |
| LogisticRegression       |   0.999242 | 0.999615 |  1        | 0.00112861  |
| LinearSVM(calibrated)    |   0.999242 | 0.999615 |  1        | 0.00114642  |
| Perceptron               |   0.998864 | 0.999422 |  0.999982 | 0.00090214  |
| Adaline_like             |   0.984091 | 0.991982 |  0.999881 | 0.0808532   |
| KNN(k=15)                |   0.986364 | 0.993119 |  0.973441 | 0.00800842  |


## Plots

- ROC curves and calibration plots are in `./plots/` (one PNG per model).
- Buffer→probability curves are also in `./plots/` (named `buffer_curve_<model>_<gender>_<age>.png`).

## Buffer → Probability table

A combined CSV is saved at `buffer_probability_table.csv` with columns: `model, gender, BQ_Age, buffer_sec, buffer_mmss, prob_accept`.

Share rows for the group your clubmates care about. At **0:00 buffer**, probabilities should be close to 50%.


## Per-model classification reports

### Perceptron
```
              precision    recall  f1-score   support

           0      0.933     1.000     0.966        42
           1      1.000     0.999     0.999      2598

    accuracy                          0.999      2640
   macro avg      0.967     0.999     0.982      2640
weighted avg      0.999     0.999     0.999      2640

```
### Adaline_like
```
              precision    recall  f1-score   support

           0      0.000     0.000     0.000        42
           1      0.984     1.000     0.992      2598

    accuracy                          0.984      2640
   macro avg      0.492     0.500     0.496      2640
weighted avg      0.968     0.984     0.976      2640

```
### LogisticRegression
```
              precision    recall  f1-score   support

           0      1.000     0.952     0.976        42
           1      0.999     1.000     1.000      2598

    accuracy                          0.999      2640
   macro avg      1.000     0.976     0.988      2640
weighted avg      0.999     0.999     0.999      2640

```
### KNN(k=15)
```
              precision    recall  f1-score   support

           0      1.000     0.143     0.250        42
           1      0.986     1.000     0.993      2598

    accuracy                          0.986      2640
   macro avg      0.993     0.571     0.622      2640
weighted avg      0.987     0.986     0.981      2640

```
### LinearSVM(calibrated)
```
              precision    recall  f1-score   support

           0      1.000     0.952     0.976        42
           1      0.999     1.000     1.000      2598

    accuracy                          0.999      2640
   macro avg      1.000     0.976     0.988      2640
weighted avg      0.999     0.999     0.999      2640

```
### DecisionTree
```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        42
           1      1.000     1.000     1.000      2598

    accuracy                          1.000      2640
   macro avg      1.000     1.000     1.000      2640
weighted avg      1.000     1.000     1.000      2640

```
### RandomForest
```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        42
           1      1.000     1.000     1.000      2598

    accuracy                          1.000      2640
   macro avg      1.000     1.000     1.000      2640
weighted avg      1.000     1.000     1.000      2640

```
### RandomForest(calibrated)
```
              precision    recall  f1-score   support

           0      1.000     1.000     1.000        42
           1      1.000     1.000     1.000      2598

    accuracy                          1.000      2640
   macro avg      1.000     1.000     1.000      2640
weighted avg      1.000     1.000     1.000      2640

```