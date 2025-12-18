# 🔥 Customer Churn Prediction – End-to-End ML Project

An industry-style **end-to-end customer churn prediction system** covering the complete ML lifecycle:
data cleaning → EDA → feature engineering → model training → API deployment → dashboard → explainability.

---

## 🚀 Project Highlights

- Full ML pipeline (raw data → production)
- LightGBM model with hyperparameter tuning
- Robust preprocessing with saved encoders
- FastAPI backend for real-time predictions
- Streamlit dashboard for business users
- SHAP for global & individual model explanations
- CSV upload → churn probability output
- Resume-ready & production-aligned architecture

---

## 🧠 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **LightGBM**
- **SHAP**
- **FastAPI**
- **Streamlit**
- **Power BI**
- **Git & GitHub**

---

## 📂 Project Structure

Churn Prediction/
│
├── Data/
│ ├── Telco-Customer-Churn.csv
│ ├── cleaned_churn_data.csv
│ ├── churn_probability.csv
│ └── churn_with_clusters.csv
│
├── Notebook/
│ ├── data_cleaning.ipynb
│ ├── EDA.ipynb
│ ├── feature_importance.ipynb
│ ├── model_comparison.ipynb
│ ├── cluster.ipynb
│ └── app.py # FastAPI backend
│
├── Dashboard/
│ └── Churn Dashboard.pbix
│
├── best_lgb_model.pkl
├── encoder.pkl
├── feature_names.pkl
├── cat_cols.pkl
├── streamlit_app.py
├── requirements.txt
└── README.md


---

## ⚙️ How It Works

1. User uploads customer CSV via Streamlit
2. Data sent to FastAPI backend
3. Preprocessing applied using saved encoders
4. LightGBM predicts churn probability
5. Results returned with:
   - churn prediction
   - churn probability
   - top high-risk customers
6. SHAP explains model decisions

---

## 🧪 Run Locally

1️⃣ Install dependencies
pip install -r requirements.txt
2️⃣ Start FastAPI server
uvicorn Notebook.app:app --reload
3️⃣ Run Streamlit app
streamlit run streamlit_app.py




📊 Output

Churn probability per customer
High-risk customer identification
Downloadable prediction CSV
SHAP explainability (global + individual)

🎯 Use Cases

Telecom customer retention
Subscription churn analysis
Business decision support
ML system deployment practice

👩‍💻 Author

Shivani Jain
Aspiring Data Analyst / ML Engineer



