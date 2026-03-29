import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. Load and Prepare Data ---
df = pd.read_csv("data/diagnosed_cbc_data_v4.csv")

# Splitting features (X) and target (y)
target_column = "Diagnosis"
y_raw = df[target_column]
X = df.drop(columns=[target_column])

# Encoding labels (Text to Categorical)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y_categorical = to_categorical(y_encoded)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Feature Scaling (Normalization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 2. Build Model Architecture ---
model = Sequential([
    Dense(128, input_dim=14, activation='relu'),
    Dropout(0.2),

    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),
    Dense(16, activation='relu'),

    Dense(9, activation='softmax')
])

# --- 3. Compile and Train Model ---
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=90,
    batch_size=14,
    validation_split=0.21,
    verbose=1
)

# --- 4. Model Evaluation ---
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n" + "=" * 40)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print("=" * 40 + "\n")

# --- 5. Single Prediction Example ---
# Sample data prepared as a DataFrame to maintain feature names (prevents warnings)
sample_data = pd.DataFrame([[
    9.4, 34.0, 51.0, 4.1, 5.0, 2.62, 7.1,
    27.3, 89.2, 25.1, 31.1, 187.1, 12.7, 0.16
]], columns=X.columns)

# Scaling and Prediction
sample_scaled = scaler.transform(sample_data)
prediction_probs = model.predict(sample_scaled, verbose=0)
predicted_class_index = np.argmax(prediction_probs)

# Final Result
final_diagnosis = label_encoder.inverse_transform([predicted_class_index])
print(f"Predicted Class Index: {predicted_class_index}")
print(f"Diagnosis Result: {final_diagnosis[0]}")

model.save("models/anemia_model.keras")
joblib.dump(scaler, "models/anemia_scaler.pkl")
print("Model and Scaler have been saved to the 'models/' folder.")