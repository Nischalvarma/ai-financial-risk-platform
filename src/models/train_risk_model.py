import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

print("Fraud Risk Modeling Pipeline Started")

# Load dataset
print("Loading feature dataset")

df = pd.read_csv(
    "data/processed/customer_features.csv"
)

print(df.head())

# Create binary target
df["target"] = (
    df["fraud_ratio"] > 0.5
).astype(int)

# Features
X = df.drop(
    columns=[
        "customer_id",
        "target"
    ]
)

# Target
y = df["target"]

print("Preparing ML features")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42
)

print("=" * 60)

# ==========================================================
# Logistic Regression
# ==========================================================

print("Training Model: Logistic Regression")

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_predictions = lr_model.predict(
    X_test
)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

print(f"Accuracy: {lr_accuracy:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        lr_predictions
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        lr_predictions
    )
)

print("=" * 60)

# ==========================================================
# Random Forest
# ==========================================================

print("\nTraining Model: Random Forest")

rf_model = RandomForestClassifier(

    n_estimators=200,

    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print(f"Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)

print("=" * 60)

# ==========================================================
# XGBoost
# ==========================================================

print("\nTraining Model: XGBoost")

xgb = XGBClassifier(

    n_estimators=200,

    learning_rate=0.05,

    max_depth=6,

    random_state=42,

    eval_metric="logloss"
)

xgb.fit(
    X_train,
    y_train
)

xgb_predictions = xgb.predict(
    X_test
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

print(f"Accuracy: {xgb_accuracy:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        xgb_predictions
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        xgb_predictions
    )
)

print("=" * 60)

# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

joblib.dump(
    xgb,
    "src/models/fraud_model.pkl"
)

print("Trained XGBoost model saved.")

print("\nPipeline execution completed")