import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/customer_features.csv"


def load_data():

    print("Loading feature dataset")

    df = pd.read_csv(DATA_PATH)

    return df


def prepare_data(df):

    print("Preparing neural network features")

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

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def build_neural_network(input_dim):

    model = tf.keras.Sequential([

        tf.keras.layers.Dense(
            64,
            activation='relu',
            input_shape=(input_dim,)
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            32,
            activation='relu'
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16,
            activation='relu'
        ),

        tf.keras.layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":

    print("Deep Learning Pipeline Started")

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_neural_network(
        X_train.shape[1]
    )

    print("Training neural network")

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=30,

        batch_size=32,

        verbose=1
    )

    print("Evaluating model")

    predictions = model.predict(X_test)

    predictions = (
        predictions > 0.5
    ).astype(int)

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

    print()

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")

    model.save(
        "src/deep_learning/fraud_risk_model.keras"
    )

    print("Neural network saved successfully")

    print("Pipeline execution completed")