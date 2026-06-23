"""
Credit Default Risk — demo app (Streamlit)
==========================================
A showcase for the trained model: enter a new applicant by hand, get the
predicted default probability, the cost-based decision and a SHAP explanation.

This is a demonstrator, not a production credit system. It loads the FULL model
(credit_model.pkl) as trained — every field, including Sex and Age, feeds the
model — precisely so the app can make the fairness problem visible instead of
hiding it (see the fairness note in the app).

Run with:  streamlit run app.py
"""

import json

import matplotlib
matplotlib.use("Agg")   # headless backend, plays well with st.pyplot
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import shap
import streamlit as st


# ======================================================================
# 1. INPUT DOMAINS (must match the categories the model was trained on)
# ======================================================================
# the readable category values come straight from the training data; the account
# fields carry an explicit "missing" level, because no account info is itself a
# signal (kept as a category in training, not dropped).
SEX_OPTIONS = ["male", "female"]
JOB_OPTIONS = ["unskilled_non_resident", "unskilled_resident", "skilled", "highly_skilled"]
HOUSING_OPTIONS = ["own", "rent", "free"]
SAVING_OPTIONS = ["little", "moderate", "quite rich", "rich", "missing"]
CHECKING_OPTIONS = ["little", "moderate", "rich", "missing"]
PURPOSE_OPTIONS = ["car", "radio/TV", "furniture/equipment", "business", "education",
                   "domestic appliances", "repairs", "vacation/others"]

# the categorical columns one-hot encoded at training time, same order as §5
CATEGORICAL_COLS = ["Sex", "job_desc", "Housing", "Saving_accounts", "Checking_account", "Purpose"]

# a plausible applicant, used by the "load example" button
EXAMPLE = {
    "Age": 35, "Sex": "male", "job_desc": "skilled", "Housing": "own",
    "Saving_accounts": "little", "Checking_account": "little",
    "Credit_amount": 4000, "Duration": 24, "Purpose": "car",
}


# ======================================================================
# 2. ARTIFACTS (model, column layout, training constants)
# ======================================================================
@st.cache_resource
def load_artifacts():
    """load the saved model, its column layout, the feature constants and a SHAP
    explainer once, then reuse them across reruns."""
    model = joblib.load("credit_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    with open("feature_config.json") as f:
        config = json.load(f)
    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    explainer = shap.TreeExplainer(model)
    return model, model_columns, config, metrics, explainer


# ======================================================================
# 3. FEATURE RECONSTRUCTION (rebuild the vector exactly as in training)
# ======================================================================
def build_features(raw, model_columns, config):
    """rebuild one applicant's feature vector the same way the training pipeline
    does, then reindex onto model_columns. fails loudly if anything does not line
    up, so a silent train/serve mismatch can never reach the model."""
    missing = config["missing_account_token"]

    # base fields the model keeps as-is (the dropped re-encodings — age_band,
    # checking_missing, Job — are deliberately not rebuilt: they are not in
    # model_columns).
    row = {
        "Age": raw["Age"],
        "Credit_amount": raw["Credit_amount"],
        "Duration": raw["Duration"],
        "Sex": raw["Sex"],
        "job_desc": raw["job_desc"],
        "Housing": raw["Housing"],
        "Saving_accounts": raw["Saving_accounts"],
        "Checking_account": raw["Checking_account"],
        "Purpose": raw["Purpose"],
    }

    # engineered features: same formulas as the SQL view, plus the train-fit
    # high_amount that uses the stored 75th percentile (not a fresh one).
    row["amount_per_month"] = round(raw["Credit_amount"] / raw["Duration"], 1)
    row["long_term_loan"] = int(raw["Duration"] > config["duration_long_term_months"])
    row["high_amount"] = int(raw["Credit_amount"] > config["credit_amount_q75"])
    row["thin_file"] = int(raw["Saving_accounts"] == missing and raw["Checking_account"] == missing)

    encoded = pd.get_dummies(pd.DataFrame([row]), columns=CATEGORICAL_COLS,
                             drop_first=False).astype(float)

    # a dummy the model never saw means an input slipped through with an
    # unexpected category; reindex would drop it silently, so we stop instead.
    unknown = [c for c in encoded.columns if c not in model_columns]
    if unknown:
        raise ValueError(f"unexpected feature columns not in model_columns: {unknown}")

    aligned = encoded.reindex(columns=model_columns, fill_value=0.0)
    if list(aligned.columns) != list(model_columns):
        raise ValueError("rebuilt feature vector does not match model_columns layout")
    if aligned.isna().any().any():
        raise ValueError("rebuilt feature vector contains NaNs")
    return aligned


# ======================================================================
# 4. EXPLANATION HELPERS (turn SHAP values into readable factors)
# ======================================================================
def base_feature(col):
    """the original variable a (possibly one-hot) column belongs to."""
    for base in CATEGORICAL_COLS:
        if col.startswith(base + "_"):
            return base
    return col


def describe(base, raw):
    """a short human-readable phrase for an original variable, using the
    applicant's actual value (so we never report an inactive dummy)."""
    readable = {
        "Age": f"age {raw['Age']}",
        "Credit_amount": f"credit amount {raw['Credit_amount']:,}",
        "Duration": f"loan duration {raw['Duration']} months",
        "amount_per_month": f"monthly installment ~{round(raw['Credit_amount'] / raw['Duration'])}",
        "high_amount": "a high credit amount",
        "long_term_loan": "a long-term loan",
        "thin_file": "a thin financial profile (no account info)",
    }
    if base in readable:
        return readable[base]
    if base in CATEGORICAL_COLS:
        label = base.replace("_", " ").replace("job desc", "job")
        return f"{label}: {raw[base]}"
    return base


def top_factors(shap_row, model_columns, raw, k=4):
    """the k original variables that move this prediction the most. one-hot dummies
    are summed back into their variable first (shap is additive), so a categorical
    is reported once, by the applicant's actual value, not as scattered dummies."""
    grouped = {}
    for col, value in zip(model_columns, shap_row):
        grouped[base_feature(col)] = grouped.get(base_feature(col), 0.0) + value
    ordered = sorted(grouped.items(), key=lambda p: abs(p[1]), reverse=True)
    factors = []
    for base, value in ordered[:k]:
        direction = "raises" if value > 0 else "lowers"
        factors.append(f"{describe(base, raw)} {direction} the predicted risk")
    return factors


# ======================================================================
# 5. APP
# ======================================================================
def main():
    st.set_page_config(page_title="Credit Risk Evaluator", page_icon="💳", layout="centered")
    model, model_columns, config, metrics, explainer = load_artifacts()
    threshold = config["operating_threshold"]

    st.title("Credit risk evaluator")
    st.caption("A demo of the Credit Default Risk + XAI project — enter an applicant "
               "and see the prediction, the decision and why.")

    # fairness note (passive): the model uses protected attributes on purpose, to
    # make the issue visible. we say so up front rather than hide it.
    di = metrics.get("fairness", {}).get("disparate_impact_ratio", 0.79)
    st.warning(
        f"**Fairness note.** This demo runs the full model, which includes **Sex** and "
        f"**Age** — protected attributes. In a real credit system they should be "
        f"**excluded** for compliance (German AGG, EU Gender Directive 2004/113). The "
        f"project's fairness audit shows their use introduces discrimination: the "
        f"Disparate Impact ratio is **{di}**, which fails the 80% rule. The app keeps "
        f"them on purpose, to make that problem visible — not to endorse it."
    )

    # --- inputs -------------------------------------------------------
    for key, value in EXAMPLE.items():
        st.session_state.setdefault(key, value)
    if st.button("load example"):
        st.session_state.update(EXAMPLE)
        st.rerun()

    st.subheader("Applicant")
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Age", 19, 75, key="Age")
        st.selectbox("Sex", SEX_OPTIONS, key="Sex")
        st.selectbox("Job", JOB_OPTIONS, key="job_desc")
        st.selectbox("Housing", HOUSING_OPTIONS, key="Housing")
        st.selectbox("Purpose", PURPOSE_OPTIONS, key="Purpose")
    with col2:
        st.number_input("Credit amount", min_value=250, max_value=20000, step=50, key="Credit_amount")
        st.slider("Duration (months)", 4, 72, key="Duration")
        st.selectbox("Saving account", SAVING_OPTIONS, key="Saving_accounts")
        st.selectbox("Checking account", CHECKING_OPTIONS, key="Checking_account")

    if not st.button("Evaluate credit risk", type="primary"):
        return

    raw = {k: st.session_state[k] for k in EXAMPLE}

    # --- rebuild features and predict (fail explicitly on any mismatch) ---
    try:
        features = build_features(raw, model_columns, config)
    except ValueError as err:
        st.error(f"Feature reconstruction failed: {err}")
        st.stop()

    proba = float(model.predict_proba(features)[:, 1][0])
    approved = proba < threshold

    # --- decision -----------------------------------------------------
    st.subheader("Result")
    left, right = st.columns(2)
    left.metric("Predicted default probability", f"{proba:.1%}")
    right.metric("Decision", "Approve" if approved else "Decline",
                 help=f"cost-based threshold = {threshold:.2f} (not 0.5); "
                      f"decline when P(default) ≥ {threshold:.2f}")
    if approved:
        st.success(f"Approve — predicted risk {proba:.1%} is below the {threshold:.2f} threshold.")
    else:
        st.error(f"Decline — predicted risk {proba:.1%} is at or above the {threshold:.2f} threshold.")

    # --- local SHAP explanation --------------------------------------
    st.subheader("Why — SHAP explanation")
    explanation = explainer(features)
    shap.plots.waterfall(explanation[0], show=False)
    st.pyplot(plt.gcf())
    plt.close("all")

    st.markdown("**Main drivers for this applicant**")
    for factor in top_factors(explanation.values[0], model_columns, raw):
        st.markdown(f"- {factor}")
    st.caption("SHAP values are in log-odds space; positive values push the risk up, "
               "negative values push it down.")


if __name__ == "__main__":
    import streamlit.runtime
    if streamlit.runtime.exists():
        main()                      # lanciato da `streamlit run` -> esegui la UI
    else:
        import sys, subprocess      # lanciato con `python app.py` -> rilancia con streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

