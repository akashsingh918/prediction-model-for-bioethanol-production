import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# === Step 1: Load Data ===
df = pd.read_csv('synthetic_data.csv')
X = df[['biomass_loading', 'particle_size', 'time']].values
y = df[['glucose', 'xylose']].values

# === Step 2: Normalize ===
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# === Step 3: Split Data ===
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Step 4: Build ANN ===
model = Sequential()
model.add(Dense(8, input_dim=3, activation='sigmoid'))  # logsig ≈ sigmoid
model.add(Dense(2, activation='linear'))  # purelin ≈ linear

# === Step 5: Compile & Train ===
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), verbose=0)

# === Step 6: Evaluate ===
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_true, y_pred)
r2_glucose = r2_score(y_true[:, 0], y_pred[:, 0])
r2_xylose = r2_score(y_true[:, 1], y_pred[:, 1])

print(f"Test MSE: {mse:.4f}")
print(f"R² Glucose: {r2_glucose:.2f}, R² Xylose: {r2_xylose:.2f}")

# === Step 7: Plot ===
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
