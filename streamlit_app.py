import streamlit as st
import pandas as pd
import requests
import shap
import pickle
import matplotlib.pyplot as plt

# ================= CONFIG =================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

# ================= LOAD ARTIFACTS =================
model = pickle.load(open("best_lgb_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
cat_cols = pickle.load(open("cat_cols.pkl", "rb"))

# ================= PREPROCESS (SAME AS API) =================
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)

    df[cat_cols] = df[cat_cols].astype(str)

    cat_encoded = encoder.transform(df[cat_cols])
    cat_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    num_df = df.drop(columns=cat_cols)

    final_df = pd.concat(
        [num_df.reset_index(drop=True),
         cat_df.reset_index(drop=True)],
        axis=1
    )

    final_df = final_df.reindex(columns=feature_names, fill_value=0)

    return final_df

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload Customer CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ================= PREDICTION =================
    if st.button("🚀 Predict Churn"):
        with st.spinner("Calling model API..."):
            response = requests.post(
                f"{API_URL}/predict/",
                files={"file": uploaded_file.getvalue()}
            )

        if response.status_code == 200:
            result = response.json()

            st.subheader("📌 Summary")
            s = result["summary"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Customers", s["total_customers"])
            c2.metric("Churned", s["churn_count"])
            c3.metric("Not Churned", s["non_churn_count"])
            c4.metric("Avg Churn Prob", round(s["average_churn_probability"], 3))
            c5.metric("High Risk %", round(s["high_risk_percent"], 2))

            st.subheader("🔥 Top 10 High-Risk Customers")
            st.dataframe(pd.DataFrame(result["top_10_high_risk_customers"]))

            st.markdown(
                f"[⬇️ Download Predictions CSV]({API_URL}{result['download_endpoint']})"
            )

        else:
            st.error("FastAPI prediction failed")

    # ================= SHAP EXPLANATION =================
    st.divider()
    st.subheader("🧠 Global Model Explanation (SHAP)")

    with st.spinner("Computing SHAP values..."):
        X_encoded = preprocess_input(df)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_encoded)

        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            X_encoded,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig)

    # ================= INDIVIDUAL EXPLANATION =================
    st.subheader("🔍 Explain Individual Customer")

    idx = st.number_input(
        "Customer row index",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    single_row = preprocess_input(df.iloc[[idx]])
    shap_single = explainer.shap_values(single_row)

    fig2 = plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_single,
        single_row,
        matplotlib=True,
        show=False
    )
    st.pyplot(fig2)
