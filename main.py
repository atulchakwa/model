from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

# Load your trained average model
model_avg = joblib.load("model_avg.pkl")

app = FastAPI(title="Stock Average Price Prediction API")

# Request body
class StockRawRequest(BaseModel):
    c: float     # Close
    d: str       # Symbol
    dp: float = 0.0
    h: float     # High
    l: float     # Low
    o: float     # Open
    pc: float    # Previous close
    t: str = None  # Optional date

# Prepare features for prediction
def convert_to_model_input(data: StockRawRequest):
    # Use provided date or today
    date = pd.to_datetime(data.t) if data.t else pd.to_datetime(datetime.today().date())
    
    # Date features
    df_date = pd.DataFrame([{
        'day_of_week': date.dayofweek,
        'month': date.month,
        'quarter': date.quarter,
        'day_of_month': date.day,
        'year': date.year
    }])
    
    # Numeric features
    df_numeric = pd.DataFrame([{
        'Open': data.o,
        'High': data.h,
        'Low': data.l,
        'Close': data.c,
        'PrevClose': data.pc,
        'DailyPct': data.dp
    }])
    
    # Combine numeric + date features
    X_final = pd.concat([df_numeric.reset_index(drop=True), df_date.reset_index(drop=True)], axis=1)
    
    return X_final

# Prediction endpoint
@app.post("/predict")
def predict_stock_raw(request: StockRawRequest):
    try:
        X = convert_to_model_input(request)
        pred_avg = model_avg.predict(X)[0]
        
        return {
            "symbol": request.d,
            "date": str(pd.to_datetime(request.t) if request.t else datetime.today().date()),
            "predicted_avg": round(float(pred_avg), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
