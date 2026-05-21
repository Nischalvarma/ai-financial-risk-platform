import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from xgboost import XGBClassifier


DATA_PATH = "data/processed/customer_features.csv"


def load_data():

    print("Loading feature dataset")

    df = pd.read_csv(DATA_PATH)

    print(df.head())

    return df


def prepare_data(df):

    print("Preparing ML features")

    df['high_risk_customer'] = (
        df['fraud_ratio'] > 0.05
    ).astype(int)

    features = [

        'transaction_count',

        'avg_transaction_amount',

        'max_transaction_amount',

        'min_transaction_amount',

        'total_transaction_amount',

        'total_credit_transactions',

        'total_debit_transactions',

        'fraud_transactions',

        'fraud_ratio'
    ]

    X = df[features]

    y = df['high_risk_customer']

    return X, y


def train_models(X_train, X_test, y_train, y_test):

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )
    }

    for name, model in models.items():

        print("=" * 60)

        print(f"Training Model: {name}")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy:.4f}")

        print()

        print("Classification Report")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print()

        print("Confusion Matrix")

        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        print("=" * 60)

        print()


if __name__ == "__main__":

    print("Fraud Risk Modeling Pipeline Started")

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("Pipeline execution completed")