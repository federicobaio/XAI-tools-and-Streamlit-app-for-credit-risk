"""
Credit Default Risk — Modeling & Explainability (XAI)
=====================================================
The base features are engineered upstream in SQL on BigQuery (see
query_credit_risk.sql) and exported to `credit_features.csv`. This script
consumes that table, then performs the leakage-sensitive steps (encoding and
the two cross-row statistical features) inside the train/test boundary.

Author: Federico Baio
"""

import json
import os
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

# silence only the noisy version/deprecation warnings from sklearn/shap/lime,
# so real ones (e.g. RuntimeWarning) stay visible.
for _cat in (FutureWarning, DeprecationWarning, UserWarning):
    warnings.filterwarnings("ignore", category=_cat)
RANDOM_STATE = 42
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Cross-validation scheme reused everywhere: shuffled + stratified.
# (Shuffling is essential because the dataset is ordered by Age)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ======================================================================
# 1. LOAD DATA
# ======================================================================

df = pd.read_csv("credit_features.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
print("Target distribution:\n", df["default_flag"].value_counts(), "\n")

# ======================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ======================================================================

# 2a. Default rate by loan purpose
purpose_rate = (df.groupby("Purpose")["default_flag"].mean().sort_values(ascending=False) * 100)
purpose_n = df["Purpose"].value_counts().reindex(purpose_rate.index)
ax = purpose_rate.plot(kind="barh", color="#c0392b", figsize=(7, 4))
ax.set_xlabel("Default rate (%)")
ax.set_title("Default rate by loan purpose")
# annotate sample size: rates for small categories (e.g. vacation/others) are noisy
for i, n in enumerate(purpose_n.values):
    ax.text(purpose_rate.iloc[i] + 0.3, i, f"n={n}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_default_by_purpose.png", dpi=120)
plt.close()

# 2b. Default rate by credit-amount band. raw frequency histograms are dominated by
# the 70/30 class imbalance, so instead we bin the amount into equal-size quantile
# bands and show the default rate in each -> a direct, undistorted read of whether
# bigger loans default more.
amount_band = pd.qcut(df["Credit_amount"], q=5)
rate_by_amount = df.groupby(amount_band, observed=True)["default_flag"].mean() * 100
ax = rate_by_amount.plot(kind="bar", color="#c0392b", figsize=(7, 4))
ax.set_xlabel("Credit amount band (equal-size quantile bins)")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate by credit amount band")
ax.axhline(df["default_flag"].mean() * 100, color="grey", linestyle="--",
           label=f"overall = {df['default_flag'].mean() * 100:.0f}%")
ax.legend()
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/eda_default_by_amount.png", dpi=120)
plt.close()


# ======================================================================
# 3. PREPROCESSING
# ======================================================================

y = df["default_flag"]
X = df.drop(columns=["default_flag"])

# Missing values in the two account columns are informative ("no account info"),
# so we keep them as an explicit category rather than dropping rows.
for col in ["Saving_accounts", "Checking_account"]:
    X[col] = X[col].fillna("missing")

# --- Drop redundant re-encodings to avoid collinearity -----------------
#   age_band         -> redundant with the continuous `Age` (binning loses info,
#                       and SHAP dependence is more informative on the raw value)
#   Job (numeric)    -> kept as the readable one-hot `job_desc` instead; the 0-3
#                       code is not cleanly ordinal, so a nominal encoding is more correct
#   checking_missing -> already captured by the `Checking_account_missing` dummy
X = X.drop(columns=["age_band", "Job", "checking_missing"])

# --- Drop the two SQL features that leak test information ---------------
# `amount_vs_purpose_avg` (deviation from the per-Purpose mean credit amount) and
# `high_amount` (above the 75th amount percentile) were computed in SQL over the
# WHOLE table, i.e. using rows that later land in the test set. We recompute them
# below using TRAIN statistics only.
X = X.drop(columns=["amount_vs_purpose_avg", "high_amount"])


# ======================================================================
# 4. TRAIN / TEST SPLIT (preserving the 70/30 class balance)
# ======================================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"Train: {X_train_raw.shape[0]} rows  Test: {X_test_raw.shape[0]} rows\n")

# Keep the protected attribute aside (indexed) for the fairness audit in section 13.
sex_test = X_test_raw["Sex"].copy()


# ======================================================================
# 5. FEATURE ENGINEERING (features on TRAIN, applied also to TEST)
# ======================================================================
def add_train_fit_features(train_raw, test_raw):
    """Recompute the two cross-row features using training statistics only."""
    tr, te = train_raw.copy(), test_raw.copy()

    # deviation of the credit amount from its per-Purpose average
    purpose_mean = tr.groupby("Purpose")["Credit_amount"].mean()
    global_mean = tr["Credit_amount"].mean()
    for frame in (tr, te):
        mapped = frame["Purpose"].map(purpose_mean).fillna(global_mean)
        frame["amount_vs_purpose_avg"] = (frame["Credit_amount"] - mapped).round(0)

    # "high amount" flag relative to the train 75th percentile
    q75 = tr["Credit_amount"].quantile(0.75)
    for frame in (tr, te):
        frame["high_amount"] = (frame["Credit_amount"] > q75).astype(int)

    return tr, te


X_train_raw, X_test_raw = add_train_fit_features(X_train_raw, X_test_raw)

# One-hot encode the text columns. Categories are learned on the TRAIN frame, and
# the test frame is reindexed based on the train columns (unseen categories -> all
# zeros, missing columns -> 0) so the two matrices line up exactly.
categorical_cols = X_train_raw.select_dtypes(include=["object", "string"]).columns.tolist()
X_train = pd.get_dummies(X_train_raw, columns=categorical_cols, drop_first=False).astype(float)
X_test = (pd.get_dummies(X_test_raw, columns=categorical_cols, drop_first=False)
            .astype(float)
            .reindex(columns=X_train.columns, fill_value=0.0))
print(f"Features after encoding: {X_train.shape[1]} columns "
      f"({len(categorical_cols)} categoricals one-hot encoded).\n")


# ======================================================================
# 6. MULTICOLLINEARITY CHECK (VIF) — keep the explanations faithful
# ======================================================================
# Several engineered features are derived from Credit_amount, so they are highly
# correlated. Collinearity does not hurt the predictive accuracy of tree ensembles,
# but it does corrupt the explanations: SHAP splits one real effect arbitrarily
# across the correlated proxies, and LIME's independent perturbation generates
# impossible points. Since this project is about explainability, we measure the
# collinearity with the Variance Inflation Factor (VIF = 1/(1-R^2)) and drop the
# redundant derived features, keeping the readable base feature Credit_amount.
def compute_vif(frame, cols):
    vif = {}
    for col in cols:
        others = [c for c in cols if c != col]
        r2 = LinearRegression().fit(frame[others], frame[col]).score(frame[others], frame[col])
        vif[col] = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")
    return pd.Series(vif).sort_values(ascending=False)

continuous_cols = ["Age", "Credit_amount", "Duration",
                   "amount_per_month", "amount_per_age", "amount_vs_purpose_avg"]
print(compute_vif(X_train, continuous_cols).round(2).to_string(), "\n")

# amount_vs_purpose_avg (~0.96 corr with Credit_amount) and amount_per_age (~0.91)
# are near-duplicates of Credit_amount (VIF > 10). We drop them. amount_per_month
# is KEPT (VIF < 3): dividing by Duration de-correlates it, so it carries genuinely
# independent signal (loan amount per unit of time).
collinear_to_drop = ["amount_vs_purpose_avg", "amount_per_age"]
X_train = X_train.drop(columns=collinear_to_drop)
X_test = X_test.drop(columns=collinear_to_drop)

remaining_cont = [c for c in continuous_cols if c not in collinear_to_drop]
print(compute_vif(X_train, remaining_cont).round(2).to_string(), "\n")


# ======================================================================
# 7. MODEL SELECTION (cross-validated comparison on the training set)
# ======================================================================
# Logistic Regression is scaled (StandardScaler) and serves as an interpretable
# baseline; tree ensembles are the candidate black-box models.
# NOTE: the train-fit feature above (high_amount) is
# fit on the full training set, so the CV folds below share a small amount of in-fold information.
# The proper fix would be to wrap the feature engineering inside each fold; we
# keep it explicit here and do not do it since: 1) do not touch the test set,
# but only influence the value for the comparison of the models, and 2) the difference is
# not that big to justify a heavier CV

# on imbalance: LR and RF get class_weight="balanced"; GradientBoosting has no such
# option in sklearn, but we don't need it -> the 70/30 imbalance is handled once,
# downstream, by the cost-based threshold (section 10). adding weights to the model
# too would double-count the same thing and slightly lower the auc.
candidates = {"LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)}

cv_results = {}
for name, clf in candidates.items():
    scores = cross_val_score(clf, X_train, y_train, cv=CV, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"  {name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
print()


# ======================================================================
# 8. HYPERPARAMETER TUNING (gradient boosting)
# ======================================================================

param_grid = {"n_estimators": [100, 200, 400],
    "max_depth": [2, 3],
    "learning_rate": [0.02, 0.03, 0.05, 0.1]}

grid = GridSearchCV(GradientBoostingClassifier(random_state=RANDOM_STATE), param_grid, cv=CV, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
print(f"Best params: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.3f}\n")


# ======================================================================
# 9. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ======================================================================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["good (0)", "bad (1)"]))
test_auc = roc_auc_score(y_test, y_proba)
print(f"Test ROC-AUC: {test_auc:.3f}\n")

# ROC and Precision-Recall curves (PR is informative under class imbalance)
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC curve — credit default model")
plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=120, bbox_inches="tight")
plt.close()

PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall curve — credit default model")
plt.savefig(f"{FIG_DIR}/pr_curve.png", dpi=120, bbox_inches="tight")
plt.close()


# ======================================================================
# 10. COST-BASED DECISION THRESHOLD
# ======================================================================
# In credit, a false negative (lending to a defaulter -> lost capital) costs far
# more than a false positive (rejecting a good client -> lost revenue). We pick
# the threshold that minimizes expected cost instead of defaulting to 0.5.
#
# Crucially, the threshold is selected on out-of-fold predictions of the TRAIN
# set (cross_val_predict). 

COST_FN, COST_FP = 5, 1
thresholds = np.linspace(0.05, 0.95, 91)

oof_proba = cross_val_predict(clone(model), X_train, y_train, cv=CV, method="predict_proba")[:, 1]
oof_costs = []
for t in thresholds:
    tn, fp, fn, tp = confusion_matrix(y_train, (oof_proba >= t).astype(int)).ravel()
    oof_costs.append(fn * COST_FN + fp * COST_FP)

# The cost curve is flat near its minimum, so the absolute argmin is unstable
# (driven by sampling noise). Following a "tolerance" rule, we keep the HIGHEST
# threshold whose OOF cost stays within 10% of the minimum: in that flat region
# the recall on defaulters is essentially unchanged, but a higher threshold
# approves more good clients (far fewer false positives): a more robust and
# business-sensible choice than the raw minimum.
COST_TOLERANCE = 1.10
min_cost = min(oof_costs)
eligible = [t for t, c in zip(thresholds, oof_costs) if c <= min_cost * COST_TOLERANCE]
best_t = float(max(eligible))

for t in [0.5, best_t]:
    tn, fp, fn, tp = confusion_matrix(y_test, (y_proba >= t).astype(int)).ravel()
    recall_bad = tp / (tp + fn)
    print(f"  threshold {t:.2f} -> recall(bad)={recall_bad:.2f}  "
          f"FN={fn}  FP={fp}  test_cost={fn*COST_FN+fp*COST_FP}")
print(f"Cost-minimizing threshold (FN costs {COST_FN}x FP): {best_t:.2f}\n")

# the report in section 9 is at 0.5; this is the one that matters in practice,
# at the operating threshold we would actually deploy.
print(classification_report(y_test, (y_proba >= best_t).astype(int),
                            target_names=["good (0)", "bad (1)"]))

# plot cost vs threshold (out-of-fold curve used to make the decision)
plt.figure(figsize=(7, 4))
plt.plot(thresholds, oof_costs, color="#2c3e50")
plt.axvline(best_t, color="#c0392b", linestyle="--", label=f"chosen (tolerance) = {best_t:.2f}")
plt.axvline(0.5, color="grey", linestyle=":", label="default = 0.50")
plt.xlabel("Decision threshold")
plt.ylabel(f"Expected cost (FN={COST_FN}, FP={COST_FP})")
plt.title("Out-of-fold cost vs decision threshold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/threshold_cost.png", dpi=120)
plt.close()

# sensitivity: the threshold rests on two judgement calls (the cost ratio and the
# tolerance margin), so we show how it reacts to both of them
def optimal_threshold(cfn, cfp, tol):
    costs = [(lambda c: c[2] * cfn + c[1] * cfp)(
        confusion_matrix(y_train, (oof_proba >= t).astype(int)).ravel()) for t in thresholds]
    return max(t for t, c in zip(thresholds, costs) if c <= min(costs) * tol)

ratios, ratio_thr = [2, 3, 5, 10], []
for r in ratios:
    t = optimal_threshold(r, 1, COST_TOLERANCE)
    tn, fp, fn, tp = confusion_matrix(y_test, (y_proba >= t).astype(int)).ravel()
    ratio_thr.append(t)
    print(f"  FN:FP = {r:2d}:1 -> threshold {t:.2f}  (test recall_bad={tp/(tp+fn):.2f}, FP={fp})")

for tol in [1.00, 1.05, 1.10, 1.20]:
    print(f"  tolerance {tol:.2f} -> threshold {optimal_threshold(COST_FN, COST_FP, tol):.2f}")
print()

# figure: optimal threshold vs cost ratio
plt.figure(figsize=(7, 4))
plt.plot(ratios, ratio_thr, marker="o", color="#2c3e50")
plt.axhline(best_t, color="#c0392b", linestyle="--",
            label=f"chosen ({COST_FN}:{COST_FP}) = {best_t:.2f}")
plt.xlabel("cost ratio FN:FP")
plt.ylabel("cost-optimal threshold")
plt.title("How the decision threshold moves with the cost ratio")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/threshold_sensitivity.png", dpi=120)
plt.close()


# ======================================================================
# 11. EXPLAINABILITY: SHAP (global + local)
# ======================================================================

explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)   # note: values are in log-odds (margin) space
mean_abs_shap = pd.Series(np.abs(explanation.values).mean(axis=0), index=X_test.columns)

# 11a. Global beeswarm: how each feature pushes predictions across all clients.
# (the "no checking account" dummy pushes risk DOWN, which looks backwards but is a
# documented german-credit pattern: clients with no checking record repay better.)
shap.summary_plot(explanation.values, X_test, show=False, max_display=15)
plt.title("SHAP — global feature importance (beeswarm)")
plt.savefig(f"{FIG_DIR}/shap_beeswarm.png", dpi=120, bbox_inches="tight")
plt.close()

# 11b. Global bar: mean absolute SHAP value per (encoded) feature
shap.summary_plot(explanation.values, X_test, plot_type="bar",
                  show=False, max_display=15)
plt.title("SHAP — mean impact per feature")
plt.savefig(f"{FIG_DIR}/shap_bar.png", dpi=120, bbox_inches="tight")
plt.close()

# 11c. Grouped importance: one-hot splits a variable across several dummies. shap is
# additive, so the honest per-variable importance is mean|sum of the dummy
# contributions per client| -> we sum the dummies within each client first, then take
# the mean abs. (summing each dummy's mean|shap| instead would overstate categoricals
# vs the single-column continuous features, since the dummies' signs can't cancel.)
def base_feature(col):
    for c in categorical_cols:
        if col.startswith(c + "_"):
            return c
    return col
shap_by_feature = (pd.DataFrame(explanation.values.T, index=X_test.columns)
                   .groupby(base_feature).sum())   # sum dummy contributions per client
grouped_shap = shap_by_feature.abs().mean(axis=1).sort_values(ascending=False)
grouped_shap.head(12).iloc[::-1].plot(kind="barh", color="#2c3e50", figsize=(7, 5))
plt.xlabel("mean |summed SHAP over a variable's dummies| (log-odds)")
plt.title("SHAP — importance grouped by original feature")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_grouped.png", dpi=120, bbox_inches="tight")
plt.close()
# checking_account dominates (a known german-credit pattern). of the SQL-engineered
# features, amount_per_month and job_desc carry real weight; high_amount, thin_file
# and long_term_loan contribute ~0 -> they could be pruned, but we keep them as
# readable business flags (still useful info even if the model does not use them).

# 11d. Dependence: relate a feature's value to its SHAP impact. a dependence plot is
# pointless on a binary feature (just two stripes), so we pick the strongest
# CONTINUOUS feature rather than the top one overall (which is binary).
continuous = [c for c in X_test.columns if X_test[c].nunique() > 2]
dep_feature = mean_abs_shap[continuous].idxmax()
shap.dependence_plot(dep_feature, explanation.values, X_test,
                     show=False, interaction_index=None)
plt.title(f"SHAP dependence — {dep_feature}")
plt.savefig(f"{FIG_DIR}/shap_dependence.png", dpi=120, bbox_inches="tight")
plt.close()
# note: for amount_per_month the relation is inverse (low installment -> higher risk).
# it makes sense: a low monthly installment usually means a long-duration loan, which
# is the riskier profile -> amount_per_month acts partly as a proxy for Duration.

# 11e. Local waterfall: why the single riskiest applicant was flagged
# `idx` is the position of the riskiest client inside the test arrays (used to
# index SHAP/LIME); `applicant_id` is that client's real row in the source table.
idx = int(y_proba.argmax())
applicant_id = int(X_test.index[idx])
print(f"Explaining applicant #{applicant_id} (test position {idx}): "
      f" predicted default risk: {y_proba[idx]:.2f}")
shap.plots.waterfall(explanation[idx], max_display=12, show=False)
plt.title(f"SHAP — why applicant #{applicant_id} is high-risk")
plt.savefig(f"{FIG_DIR}/shap_waterfall.png", dpi=120, bbox_inches="tight")
plt.close()


# ======================================================================
# 12. EXPLAINABILITY: LIME (local)
# ======================================================================

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=["good", "bad"],
    mode="classification",
    random_state=RANDOM_STATE)

lime_exp = lime_explainer.explain_instance(
    data_row=X_test.values[idx],
    predict_fn=model.predict_proba,
    num_features=10)

lime_exp.save_to_file(f"{FIG_DIR}/lime_explanation.html")
fig = lime_exp.as_pyplot_figure()
fig.set_size_inches(8, 5)
plt.title(f"LIME — local explanation for applicant #{applicant_id}")
plt.tight_layout()
fig.savefig(f"{FIG_DIR}/lime_explanation.png", dpi=120)
plt.close(fig)

# cross-check: do SHAP and LIME tell the same story for this applicant? SHAP is in
# log-odds and LIME in probability, so we compare only what is scale-free -> the
# sign of each contribution (direction) and the rank of their magnitudes (Spearman).
lime_weights = {X_train.columns[i]: w for i, w in lime_exp.as_map()[1]}
shap_inst = pd.Series(explanation.values[idx], index=X_test.columns)
sign_agree = int(sum(np.sign(shap_inst[f]) == np.sign(w) for f, w in lime_weights.items()))
rank_corr = spearmanr([abs(shap_inst[f]) for f in lime_weights],
                      [abs(w) for w in lime_weights.values()]).correlation
print(f"SHAP vs LIME on applicant #{applicant_id}: "
      f"sign agreement {sign_agree}/{len(lime_weights)}, rank corr {rank_corr:.2f}")


# ======================================================================
# 13. FAIRNESS AUDIT — does the model treat a protected attribute fairly?
# ======================================================================
# `Sex` and `Age` are protected attributes. no statute bans a column outright, but
# using them in credit decisions raises discrimination risk under German anti-
# discrimination law (AGG) and the EU Gender Directive 2004/113, so best practice
# excludes them. we keep `Sex` in the model here only to audit how much it
# influences predictions; the explainability tooling makes this
# measurable rather than a guess. Fairness is multi-dimensional, so we look at it
# from four angles instead of trusting a single number:
#   (a) base rate     : do the groups actually default at different rates?
#   (b) demographic parity (Disparate Impact): are approval rates similar?
#   (c) equal opportunity: among clients with the same true outcome, are error
#                          rates similar?
#   (d) SHAP reliance : how important is Sex relative to legitimate features?
sex_cols = [c for c in X_test.columns if c.startswith("Sex_")]
decisions = pd.DataFrame({
    "Sex": sex_test.values,
    "approved": (y_proba < best_t).astype(int),
    "y_true": y_test.values,
}, index=sex_test.index)


# (a) Base rate: the groups genuinely differ, so "fair" is not obvious.
base_rate = df.groupby("Sex")["default_flag"].mean()
for sex, r in base_rate.items():
    print(f"  ACTUAL default rate ({sex:7s}): {r:.1%}")

# (b) Demographic parity / Disparate Impact (the 80% rule).
approval_by_sex = decisions.groupby("Sex")["approved"].mean()
di_ratio = approval_by_sex.min() / approval_by_sex.max()
for sex, rate in approval_by_sex.items():
    print(f"  approval rate ({sex:7s}): {rate:.2%}")
print(f"  Disparate Impact ratio: {di_ratio:.2f} "
      f"({'PASS' if di_ratio >= 0.8 else 'FAIL'} the 80% rule)")

# (c) Equal opportunity, split by the true outcome. The key harm is the
# false-rejection rate among clients who would actually have repaid (y_true==0).
good = decisions[decisions["y_true"] == 0]
false_reject = (1 - good.groupby("Sex")["approved"].mean())  # rejected good clients
eo_gap = float(false_reject.max() - false_reject.min())
for sex, r in false_reject.items():
    print(f"  false-rejection rate among GOOD clients ({sex:7s}): {r:.2%}")
print(f"  Equal-opportunity gap (false-rejection of good clients): {eo_gap:.2f}")

# (d) How much the model relies on Sex internally (grouped per-variable SHAP, §11c)
sex_shap = float(grouped_shap.get("Sex", 0.0))
sex_rank = int(grouped_shap.sort_values(ascending=False).index.get_indexer(["Sex"])[0]) + 1
print(f"  Sex grouped mean|SHAP| share: {sex_shap / grouped_shap.sum():.1%} of total "
      f"impact (Sex ranks #{sex_rank} of {len(grouped_shap)} original features)")

# Small-sample caveat: the test set is small, so these group rates are noisy.
sex_counts = decisions["Sex"].value_counts().to_dict()
print(f"  test-set group sizes: {sex_counts}")


# ----------------------------------------------------------------------
# 13b. MITIGATION EXPERIMENT: retrain without the protected attributes
# ----------------------------------------------------------------------
# "Fairness through unawareness": drop Sex (and Age) and re-audit. This also
# probes how the model used Sex: its SHAP share is tiny, yet removing it still
# moves the outcomes -> the model exploited Sex mainly through interactions with
# other features, not as a standalone effect. We report all three axes (DI,
# equal-opportunity gap, AUC) so the full picture is visible rather than a single
# fairness number.
# (Theory note - the fairness "impossibility": the groups have DIFFERENT real
# default rates (women 35% vs men 28%), and when base rates differ you cannot make
# BOTH the approval rates equal (demographic parity) AND the error rates equal
# (equal opportunity) at the same time - enforcing one unbalances the other. Which
# to prioritise is a policy choice, not a technical one. This run does not show
# the trade-off only because the small test set is noisy.)
def evaluate_scenario(drop_cols):
    cols = [c for c in X_train.columns if c not in drop_cols]
    m = clone(model)
    m.fit(X_train[cols], y_train)
    oof = cross_val_predict(m, X_train[cols], y_train, cv=CV, method="predict_proba")[:, 1]
    costs = [confusion_matrix(y_train, (oof >= t).astype(int)).ravel() for t in thresholds]
    costs = [fn * COST_FN + fp * COST_FP for (tn, fp, fn, tp) in costs]
    thr = max(t for t, c in zip(thresholds, costs) if c <= min(costs) * COST_TOLERANCE)
    proba = m.predict_proba(X_test[cols])[:, 1]
    appr = pd.Series((proba < thr).astype(int), index=X_test.index)
    rate = appr.groupby(sex_test).mean()
    di = rate.min() / rate.max()
    g = (y_test.values == 0)
    fr = (1 - appr[g].groupby(sex_test[g]).mean())
    return {"di": float(di), "eo_gap": float(fr.max() - fr.min()),
            "auc": roc_auc_score(y_test, proba)}

age_cols = [c for c in X_train.columns if c == "Age"]
scenarios = {
    "with Sex+Age (baseline)": {"di": float(di_ratio), "eo_gap": eo_gap, "auc": test_auc},
    "without Sex":             evaluate_scenario(sex_cols),
    "without Sex+Age":         evaluate_scenario(sex_cols + age_cols),}

print(f"  {'scenario':26s} {'DI':>5s}  {'EO-gap':>7s}  {'test-AUC':>8s}")
for name, m in scenarios.items():
    print(f"  {name:26s} {m['di']:5.2f}  {m['eo_gap']:7.2f}  {m['auc']:8.3f}")

# Recommendation: a deployable model should be retrained without Sex/Age. Here that
# nearly closes the gap (DI 0.79 -> 0.95). Note this is NOT data leakage: the other
# features are legitimate to collect, they are just statistically correlated with
# sex (proxy effects), so they carry a little sex-related signal. Any small residual
# disparity may be sampling noise or mild proxy correlation; which fairness metric
# to prioritise is ultimately a policy choice.

# Figure: predicted-risk distribution by sex (visual fairness check)
fig, ax = plt.subplots(figsize=(7, 4))
for sex, color in zip(sorted(decisions["Sex"].unique()), ["#2980b9", "#c0392b"]):
    mask = decisions["Sex"].values == sex
    ax.hist(y_proba[mask], bins=20, alpha=0.6, color=color, label=sex, density=True)
ax.axvline(best_t, color="grey", linestyle="--", label=f"threshold = {best_t:.2f}")
ax.set_xlabel("Predicted default risk")
ax.set_ylabel("Density")
ax.set_title("Predicted risk distribution by Sex (fairness check)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fairness_risk_by_sex.png", dpi=120)
plt.close()


# ======================================================================
# 14. SAVE ARTIFACTS (model, column layout, metrics summary)
# ======================================================================
joblib.dump(model, "credit_model.pkl")
joblib.dump(list(X_train.columns), "model_columns.pkl")

# constants the demo app needs to rebuild a new applicant's engineered features
# exactly as in training. only `credit_amount_q75` is fitted on the train set (the
# `high_amount` threshold); the rest mirror the SQL recipe in query_credit_risk.sql
# so the app can reproduce the features from raw inputs.
feature_config = {
    "credit_amount_q75": round(float(X_train_raw["Credit_amount"].quantile(0.75)), 1),
    "operating_threshold": round(best_t, 2),
    "duration_long_term_months": 24,
    "missing_account_token": "missing",
    "job_desc_map": {"0": "unskilled_non_resident", "1": "unskilled_resident",
                     "2": "skilled", "3": "highly_skilled"},
}
with open("feature_config.json", "w") as f:
    json.dump(feature_config, f, indent=2)

metrics = {
    "cv_roc_auc": {k: round(v[0], 3) for k, v in cv_results.items()},
    "best_params": grid.best_params_,
    "test_roc_auc": round(test_auc, 3),
    "optimal_threshold": round(best_t, 2),
    "riskiest_applicant_id": applicant_id,
    "riskiest_applicant_risk": round(float(y_proba[idx]), 2),
    "fairness": {
        "base_default_rate_by_sex": {k: round(float(v), 3) for k, v in base_rate.items()},
        "approval_rate_by_sex": {k: round(float(v), 3) for k, v in approval_by_sex.items()},
        "disparate_impact_ratio": round(float(di_ratio), 2),
        "equal_opportunity_gap": round(eo_gap, 2),
        "sex_shap_share": round(sex_shap / float(grouped_shap.sum()), 3),
        "mitigation": {name: {"di": round(m["di"], 2), "eo_gap": round(m["eo_gap"], 2),
                              "auc": round(float(m["auc"]), 3)}
                       for name, m in scenarios.items()}}}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

