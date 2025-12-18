from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import pickle

app = FastAPI()

# ================= LOAD ARTIFACTS =================
model = pickle.load(open("../best_lgb_model.pkl", "rb"))
encoder = pickle.load(open("../encoder.pkl", "rb"))
feature_names = pickle.load(open("../feature_names.pkl", "rb"))
cat_cols = pickle.load(open("../cat_cols.pkl", "rb"))

# ================= PREPROCESS =================
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

    # 🔥 CRITICAL: align with training
    final_df = final_df.reindex(columns=feature_names, fill_value=0)

    return final_df

# ================= API =================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = preprocess_input(df)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df["churn_probability"] = probs
    df["churn_prediction"] = preds
    df.to_csv("predictions.csv", index=False)

    summary = {
        "total_customers": len(df),
        "churn_count": int(preds.sum()),
        "non_churn_count": int(len(preds) - preds.sum()),
        "average_churn_probability": float(probs.mean()),
        "high_risk_percent": float((probs > 0.7).mean() * 100)
    }

    top_10 = (
        df[["churn_probability"]]
        .assign(row=df.index)
        .sort_values("churn_probability", ascending=False)
        .head(10)
        .to_dict(orient="records)
    )

    return {
        "summary": summary,
        "top_10_high_risk_customers": top_10,
        "download_endpoint": "/download"
    }

@app.get("/download")
def download_predictions():
    return FileResponse(
        "predictions.csv",
        filename="churn_predictions.csv"
    )

