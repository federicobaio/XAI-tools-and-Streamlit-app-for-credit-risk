# Credit Default Risk — Modeling & Explainability (XAI)

This project predicts whether a loan applicant will default, and — more importantly for me — it explains *why* the model makes each call. The dataset is the German Credit data (~1000 applicants, ~70/30 good/bad split). I treat it as a small, slightly imbalanced, real-world-flavoured problem: not a Kaggle leaderboard, but the kind of decision a lender actually has to defend.

The whole thing has three pieces:

- a SQL layer that builds the feature table on BigQuery (`query_credit_risk.sql`),
- a Python pipeline that trains, evaluates, explains and audits the model (`XAI case.py`, also as a narrated notebook `XAI case.ipynb`),
- a small Streamlit app that lets you score a new applicant by hand and see the explanation (`app.py`).

The focus is explainability and fairness, not squeezing the last point of AUC. A credit model that you cannot explain is not deployable, regardless of how accurate it is.

## The data

The base features are engineered upstream in SQL (see `query_credit_risk.sql`) and exported to `credit_features.csv`. The SQL view also keeps a few readable, redundant columns (`age_band`, `job_desc`, `checking_missing`) that are convenient for BI but would only add collinearity to a model, so the Python side keeps one encoding per signal. Two SQL columns use whole-table statistics (`amount_vs_purpose_avg`, `high_amount`); those leak test information, so the pipeline drops them and recomputes equivalents on the training set only.

## Pipeline

The script runs end to end and is split into clearly labelled sections. The decisions that matter:

- **Leakage prevention.** I split first, then fit everything (the 75th-percentile flag, the one-hot categories) on the train set only; the test set is reindexed onto the train columns. Nothing is computed using rows that later land in the test set.
- **One-hot, not label encoding.** The nominal categories (purpose, housing, accounts) have no natural order, so label encoding would invent a false one. One-hot keeps each category an independent effect.
- **Multicollinearity (VIF).** Several engineered features are derived from `Credit_amount` and end up highly correlated. This does not hurt a tree's *accuracy*, but it does corrupt the *explanations*: SHAP splits one real effect across the proxies, and LIME perturbs them independently and builds impossible points. Since this project is about explainability, I measure the Variance Inflation Factor and drop the redundant derived features, keeping the readable base ones.
- **Model selection.** Logistic Regression, Random Forest and Gradient Boosting all land around CV ROC-AUC ≈ 0.75 on this data. I keep Gradient Boosting.
- **Cost-based threshold.** A false negative (lending to a defaulter) costs far more than a false positive (rejecting a good client), so 0.5 is the wrong cutoff. I assume a 5:1 cost ratio and pick the threshold on out-of-fold predictions, taking the highest threshold whose cost stays within 10% of the minimum (the cost curve is flat near its minimum, so the exact argmin is noise). That lands at ≈ 0.28.
- **Explainability — SHAP and LIME.** SHAP (TreeExplainer, log-odds space) for global and local importance, LIME (probability space) as an independent cross-check. I compare only scale-free quantities — the sign of each contribution and the rank of the magnitudes — because the two methods live in different units. They agree on direction (9/10) and less on exact ranking (Spearman ≈ 0.55), which is the honest result.
- **Fairness audit.** A multi-axis audit on `Sex`: base rate, Disparate Impact, equal opportunity, and how much the model actually relies on `Sex` (grouped SHAP). More on this below.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Encoding | one-hot | nominal categories, no false ordering |
| Redundant derived features | dropped (via VIF) | keep SHAP/LIME faithful, accuracy unaffected |
| Final model | Gradient Boosting | best CV ROC-AUC, ties broken on simplicity |
| Decision threshold | ~0.28 (cost-based) | a missed defaulter costs ~5x a rejected good client |
| Explanation cross-check | SHAP vs LIME on sign + rank | the two live in different units; only scale-free comparison is valid |
| Deployed model | the **full** model, with `Sex`/`Age` | see fairness note — for an XAI demo, explaining the issue beats hiding it |

## Fairness — and why the deployed model still includes Sex and Age

The audit is not decorative. On the test set the model fails the 80% rule (Disparate Impact ≈ 0.79), even though `Sex` is only ~2% of the total SHAP impact. That gap flows through *proxy* variables correlated with sex, not through `Sex` directly — this is indirect discrimination, not a data leak. Retraining without the protected attributes does close most of the gap, which confirms the model was using them mainly through interactions.

So why does the demo ship the **full** model, protected attributes included? Because this is an explainability showcase, not a production lender. The point is to make the problem *visible and measurable*, not to quietly delete the awkward columns ("fairness through unawareness" is a known fallacy anyway, since the proxies remain). The app shows a clear fairness warning and the failing Disparate Impact, so the issue is front and centre.

To be explicit about the real-world side: in an actual credit system, direct use of `Sex` is heavily restricted in the EU (German AGG, Gender Directive 2004/113), and at minimum `Sex` should be excluded for compliance. `Age` is a more legitimate, commonly-used risk factor. The notebook documents the mitigation experiment for exactly this reason.

## The demo app

`app.py` is a Streamlit "credit risk evaluator". You enter an applicant by hand (every field, including `Sex` and `Age`, because it runs the full model), and it:

- rebuilds the engineered feature vector **exactly** as in training — same formulas, the stored 75th-percentile threshold, one-hot, reindex onto `model_columns.pkl` — and fails loudly if anything does not line up, so a silent train/serve mismatch can never reach the model,
- shows the predicted default probability and the decision at the cost-based threshold (not 0.5),
- shows a SHAP waterfall for that applicant and a short natural-language summary of the main drivers (one-hot dummies summed back into their original variable, so a categorical is reported once, by the applicant's actual value),
- shows the passive fairness warning described above.

The training constants the app needs (the 75th percentile, the threshold, the SQL recipe constants) are persisted to `feature_config.json` by the pipeline, so the app reproduces the features without re-reading the training code.

## How to run

```bash
pip install -r requirements.txt

# pipeline + analysis (trains the model, writes the artifacts and figures)
python "XAI case.py"      # or open XAI case.ipynb

# demo app
streamlit run app.py
```

The app opens on `http://localhost:8501`.

## Repository layout

```
XAI_case.py            pipeline + analysis (the main script)
XAI_case.ipynb         the same, as a narrated notebook
app.py                 Streamlit demo app
query_credit_risk.sql  feature engineering on BigQuery (data origin)
credit_features.csv    model input (exported from the SQL view)
credit_model.pkl       trained model
model_columns.pkl      training column layout (the serving contract)
feature_config.json    training constants the app needs to rebuild features
metrics.json           headline metrics + fairness numbers
figures/               saved plots (EDA, ROC/PR, SHAP, LIME, fairness)
lime_explanation.htlm  html output of LIME
requirements.txt       dependencies
```

## Additional information (caveats)

These are the things I would want a careful reader to know — they do not break the project, but they bound how far to trust it.

- **Small data.** ~1000 rows, and the test set has only ~200, of which ~60 are women. The fairness numbers (Disparate Impact, equal-opportunity gap) are therefore noisy; I treat them as indicative, not precise. The slight AUC change from removing the protected attributes is within that noise — I do *not* claim removal improves accuracy.
- **Residual in-fold leakage.** The train-fit statistics are computed on the whole train set, so CV folds share a sliver of information. The proper fix is to recompute them inside each fold via a `Pipeline`; the effect here is negligible and does not touch the reported test metric. I left it acknowledged rather than over-engineered.
- **SHAP vs LIME.** The agreement is on *direction* and *rank*, not on exact magnitudes — the two methods are in different units and I deliberately did not convert them. The sign match is robust but not a 100% mathematical guarantee.
- **Cost ratio.** The 5:1 ratio is a business assumption, not estimated from data. Only the ratio matters; the real cost would scale with the loan amount, which a per-client cost would capture.
- **Not a production system.** The app deploys the full model on purpose, to demonstrate the fairness audit. A real deployment would exclude `Sex` (and likely audit `Age` by bands), recompute features inside a proper serving pipeline, and monitor the disparity over time.
