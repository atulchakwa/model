from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained average model and symbol encoder
model_avg = joblib.load("model_avg.pkl")
symbol_encoder = joblib.load("symbol_encoder.pkl")

app = FastAPI(title="Stock Average Price Prediction API")

# Request body
class StockRequest(BaseModel):
    symbol: str
    date: str  # YYYY-MM-DD

# Prepare features for prediction
def prepare_features(symbol: str, date: str):
    try:
        dt = pd.to_datetime(date)
    except:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    
    # Encode symbol
    symbol_df = pd.DataFrame(symbol_encoder.transform([[symbol]]), 
                             columns=symbol_encoder.get_feature_names_out(['Symbol']))
    
    # Date features
    X = pd.DataFrame([{
        'day_of_week': dt.dayofweek,
        'month': dt.month,
        'quarter': dt.quarter,
        'day_of_month': dt.day,
        'year': dt.year
    }])
    
    # Combine with symbol one-hot
    X_final = pd.concat([X.reset_index(drop=True), symbol_df.reset_index(drop=True)], axis=1)
    return X_final

# Prediction endpoint
@app.post("/predict")
def predict_stock(request: StockRequest):
    try:
        X = prepare_features(request.symbol, request.date)
        pred_avg = model_avg.predict(X)[0]
        return {
            "symbol": request.symbol,
            "date": request.date,
            "predicted_avg": round(float(pred_avg), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
