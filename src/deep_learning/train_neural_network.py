import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv(
    "data/processed/customer_features.csv"
)

# Features
df["target"] = (df["fraud_ratio"] > 0.5).astype(int)

X = df.drop(columns=["customer_id", "target"])

y = df["target"]

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42
)

# Build neural network
model = Sequential([

    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),

    Dense(32, activation='relu'),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']
)

# Prevent overfitting
early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True
)

# Train model
model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=50,

    batch_size=32,

    callbacks=[early_stop],

    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
model.save(
    "src/deep_learning/fraud_risk_model.keras"
)

print("Neural network model saved.")