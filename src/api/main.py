import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Financial Risk Intelligence API",
    version="5.0"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(
    "src/models/fraud_model.pkl"
)

# ==========================================================
# REQUEST SCHEMA
# ==========================================================

class TransactionData(BaseModel):

    transaction_count: float

    avg_transaction_amount: float

    max_transaction_amount: float

    min_transaction_amount: float

    total_transaction_amount: float

    total_credit_transactions: float

    total_debit_transactions: float

    fraud_transactions: float

    fraud_ratio: float

# ==========================================================
# HOME ROUTE
# ==========================================================

@app.get("/")
def home():

    return {

        "message":
            "Financial Risk Intelligence API Running"
    }

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"
    }

# ==========================================================
# PREDICTION ROUTE
# ==========================================================

@app.post("/predict")
def predict_risk(data: TransactionData):

    try:

        input_data = pd.DataFrame([{

            "transaction_count":
                data.transaction_count,

            "avg_transaction_amount":
                data.avg_transaction_amount,

            "max_transaction_amount":
                data.max_transaction_amount,

            "min_transaction_amount":
                data.min_transaction_amount,

            "total_transaction_amount":
                data.total_transaction_amount,

            "total_credit_transactions":
                data.total_credit_transactions,

            "total_debit_transactions":
                data.total_debit_transactions,

            "fraud_transactions":
                data.fraud_transactions,

            "fraud_ratio":
                data.fraud_ratio
        }])

        # EXACT TRAINING ORDER
        input_data = input_data[[

            "transaction_count",

            "avg_transaction_amount",

            "max_transaction_amount",

            "min_transaction_amount",

            "total_transaction_amount",

            "total_credit_transactions",

            "total_debit_transactions",

            "fraud_transactions",

            "fraud_ratio"
        ]]

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        risk = (
            "HIGH RISK"
            if probability > 0.5
            else "LOW RISK"
        )

        return {

            "fraud_probability":
                round(probability, 4),

            "risk_level":
                risk
        }

    except Exception as e:

        return {

            "error":
                str(e)
        }