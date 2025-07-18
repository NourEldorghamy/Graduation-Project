from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict

# Load XGBoost model
model = xgb.Booster()
model.load_model('D:\\GRADUATION PROJECT\\creditCard\\fraud_xgb_model.json')

# Load label encoders
with open('D:\\GRADUATION PROJECT\\creditCard\\label_encoders.pkl', "rb") as f:
    encoders = pickle.load(f)

# FastAPI instance
app = FastAPI(title="Credit Card Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    data: List[Dict]  # list of dictionaries (records)

# Preprocessing function
def preprocess_new_data(df):
    df = df.drop(columns=['Unnamed: 0', 'cc_num', 'first', 'last', 'gender', 'street', 'zip', 'trans_num', 'unix_time'], errors='ignore')

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df['day'] = df['trans_date_trans_time'].dt.day
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

    df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
        else:
            raise ValueError(f"Missing encoder for column: {col}")

    return df

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        df_raw = pd.DataFrame(transaction.data)
        df_processed = preprocess_new_data(df_raw)

        dmatrix = xgb.DMatrix(df_processed)
        prediction_proba = model.predict(dmatrix)
        predictions = (prediction_proba >= 0.7).astype(int)

        return {
            "predictions": predictions.tolist(),
            "confidences": [round(float(p), 4) for p in prediction_proba]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
