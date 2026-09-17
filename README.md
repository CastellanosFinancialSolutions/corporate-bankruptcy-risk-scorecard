# Corporate Bankruptcy Risk Scorecard (Taiwan Stock Exchange)

A credit-risk scorecard that estimates a company's probability of
bankruptcy from its financial ratios and translates that probability into
an interpretable, auditable point score and risk grade — the same
methodology family used by FICO and by banks building PD (probability of
default) scorecards under the Basel framework.

This is a **portfolio / technical-demonstration piece**, built to show
prospective clients in credit risk, banking, and fintech an end-to-end
example of how a production-grade credit scorecard is built, validated,
and communicated — methodology, rigor, and reporting included.

## Problem

Binary classification with severe class imbalance: predict whether a
company goes bankrupt (1) or not (0) from 95 financial ratios, where only
**3.23% of cases (220 of 6,819 companies)** are actual bankruptcies. That
imbalance drives every methodological choice below (variable selection,
model weighting, metric choice, and cutoff selection) — a naive
"always predict non-bankrupt" model would already be 96.8% accurate with
zero predictive value.

## Methodology

1. **Weight of Evidence (WOE) + Information Value (IV)** for variable
   binning and selection. Of 95 original ratios, 63 passed the standard
   industry IV threshold (≥ 0.02); after removing highly correlated
   variables (> 0.85), 39 final variables were retained.
2. **Weighted logistic regression on WOE-transformed variables**, with
   stepwise AIC selection and bankruptcy cases weighted ~30x to correct
   for class imbalance.
3. **Scaling to a points-based scorecard** using the standard
   Points-to-Double-the-Odds (PDO) convention (base score 600, PDO = 50) —
   the same scaling logic used by FICO-style scorecards, fully re-calibratable
   to a client's own internal rating scale without refitting the underlying model.
4. **Random Forest run in parallel**, used exclusively as an independent
   cross-check of variable importance — not as the production model — to
   preserve the interpretability and auditability that regulated credit
   risk (Basel, IFRS 9) requires and that a black-box model does not offer
   without additional explainability tooling.

## Key results (out-of-sample test set)

| Metric | Value |
|---|---|
| AUC-ROC | 0.93 |
| KS statistic | 0.74 |
| Gini | 0.85 |
| Sensitivity (bankruptcy recall) | 81.8% |
| Specificity | 87.7% |

The resulting 5-grade risk scorecard shows a **perfectly monotonic
relationship** between grade and observed bankruptcy rate: the bottom 20%
of companies (Grade 5) concentrates **89% of all observed bankruptcies**
in the test set, while the top 3 grades (60% of companies) had **zero**
observed bankruptcies. A full discussion of results, methodology
rationale, and one explicitly documented data-scale limitation (Current
Ratio and similar variables) is in the report below.

## Data

Public dataset: UCI Machine Learning Repository, *"Taiwanese Bankruptcy
Prediction"* (id 572) — real data from the Taiwan Economic Journal, 6,819
companies listed on the Taiwan Stock Exchange, 1999-2009, 95 financial
ratios. Also mirrored on Kaggle under the same name. No private or
confidential data is involved in this case.

Source: https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction

## Repository structure

```
modelo_scorecard_quiebra.R                                        R script — full analysis pipeline
taiwan_bankruptcy_data.csv                                         source dataset (public)
Informe financiero - Riesgo de quiebra corporativa (Taiwan).md      full technical report (Spanish)
dashboard.html                                                      interactive dashboard (test-set companies)
results/                                                             output tables (IV ranking, scorecard points,
                                                                      performance metrics, risk grades, RF importance,
                                                                      test-set company scores)
plots/                                                               ROC curve, score distribution, IV ranking,
                                                                      bankruptcy rate by grade
```

## Requirements & how to run

```r
install.packages(c("here", "scorecard", "pROC", "randomForest"))
```

Clone the repo and run `modelo_scorecard_quiebra.R` from anywhere (via
`Rscript` or inside RStudio) — it locates the repo root itself with
`here::here()` (detects the cloned `.git` folder) and reads/writes
`results/` and `plots/` relative to that root. No path editing needed.

## Notes on the report language

The full technical report (`Informe financiero - Riesgo de quiebra
corporativa (Taiwan).md`) is written in Spanish — it is the detailed,
line-by-line auditable analysis. This README is an English summary for
an international audience.
