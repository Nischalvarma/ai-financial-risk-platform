import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Financial Risk Modeling Pipeline Started")

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    "data/processed/customer_features.csv"
)

print("\nDataset Preview")
print(df.head())

# ==========================================================
# CREATE BETTER FRAUD TARGET
# ==========================================================

df["target"] = (

    (
        df["fraud_ratio"] > 0.15
    )

    |

    (
        df["avg_transaction_amount"] > 20000
    )

    |

    (
        df["fraud_transactions"] > 5
    )

).astype(int)

# ==========================================================
# FEATURES
# ==========================================================

X = df[[
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

# ==========================================================
# TARGET
# ==========================================================

y = df["target"]

print("\nFraud Distribution")
print(y.value_counts())

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# ==========================================================
# TRAIN XGBOOST MODEL
# ==========================================================

print("\nTraining XGBoost Model")

model = XGBClassifier(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

joblib.dump(
    model,
    "src/models/fraud_model.pkl"
)

print("\nTrained XGBoost model saved.")

print("\nPipeline execution completed")