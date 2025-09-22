# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Step 0: Load trained model and encoder
# ----------------------------
model_avg = joblib.load("model_avg.pkl")
enc = joblib.load("symbol_encoder.pkl")  # OneHotEncoder for 'Symbol'
numeric_features = joblib.load("numeric_features.pkl")  # numeric columns used in training
feature_columns = joblib.load("feature_columns.pkl")    # X_full.columns during training

# ----------------------------
# Step 1: FastAPI app
# ----------------------------
app = FastAPI(
    title="Stock Average Price Prediction API",
    description="Predict future average stock prices",
    version="1.0"
)

# ----------------------------
# Step 2: Input data model
# ----------------------------
class StockInput(BaseModel):
    Symbol: str
    Date: str  # format: 'YYYY-MM-DD'
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float

class StockBatchInput(BaseModel):
    data: List[StockInput]

# ----------------------------
# Step 3: Prepare features
# ----------------------------
def prepare_features(df: pd.DataFrame):
    # Date features
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_month'] = df['Date'].dt.day
    df['year'] = df['Date'].dt.year

    # Numeric features
    X_num = df[numeric_features].copy()

    # Encode Symbol
    symbol_encoded = enc.transform(df[['Symbol']])
    symbol_df = pd.DataFrame(symbol_encoded, columns=enc.get_feature_names_out(['Symbol']))
    symbol_df.index = df.index

    # Combine features
    X_full = pd.concat([X_num, symbol_df], axis=1)

    # Align with training columns
    X_full = X_full.reindex(columns=feature_columns, fill_value=0)
    return X_full

# ----------------------------
# Step 4: Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict_stock(batch_input: StockBatchInput):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([item.dict() for item in batch_input.data])

        # Prepare features
        X = prepare_features(input_df)

        # Predict average price
        pred_avg = model_avg.predict(X)

        # Return predictions
        results = []
        for i in range(len(input_df)):
            results.append({
                "Symbol": input_df.loc[i, "Symbol"],
                "Date": input_df.loc[i, "Date"].strftime('%Y-%m-%d'),
                "pred_avg": float(pred_avg[i])
            })
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# Step 5: Root endpoint
# ----------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the Stock Average Price Prediction API!"}
