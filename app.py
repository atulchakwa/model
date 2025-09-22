from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

# Load the trained average model and symbol encoder
model_avg = joblib.load("model_avg.pkl")
symbol_encoder = joblib.load("symbol_encoder.pkl")

app = FastAPI(title="Stock Average Price Prediction API")

# Request body only needs symbol
class StockRequest(BaseModel):
    symbol: str

# Prepare features for prediction using today's date
def prepare_features(symbol: str):
    today = pd.to_datetime(datetime.today().date())  # today's date
    
    # Encode symbol
    symbol_df = pd.DataFrame(symbol_encoder.transform([[symbol]]), 
                             columns=symbol_encoder.get_feature_names_out(['Symbol']))
    
    # Date features
    X = pd.DataFrame([{
        'day_of_week': today.dayofweek,
        'month': today.month,
        'quarter': today.quarter,
        'day_of_month': today.day,
        'year': today.year
    }])
    
    # Combine with symbol one-hot
    X_final = pd.concat([X.reset_index(drop=True), symbol_df.reset_index(drop=True)], axis=1)
    return X_final

# Prediction endpoint
@app.post("/predict")
def predict_stock(request: StockRequest):
    try:
        X = prepare_features(request.symbol)
        pred_avg = model_avg.predict(X)[0]
        return {
            "symbol": request.symbol,
            "date": str(datetime.today().date()),  # return today’s date
            "predicted_avg": round(float(pred_avg), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
