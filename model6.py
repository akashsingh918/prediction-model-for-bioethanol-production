
#MODEL 6
# checking the robustness of the two models wiht different noise levels 
# Source: https://doi.org/10.3389/fceng.2022.994428 
# Paper: Predicting xylose yield from prehydrolysis of hardwoods: A machine learning approach

# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredError
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 📤 Upload dataset
from google.colab import files
uploaded = files.upload()

# 📄 Load and clean data
df = pd.read_csv("MonomericDataset.csv")
df = df.dropna(axis=1, how='all')
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

# ✅ Define input/output
input_features = [
    'Temp', 'TotalT', 'CA', 'Size', 'LSR',
    'F_X', 'logRo', 'logP', 'logH',
    'eucalyptus', 'aspen', 'oak', 'maple', 'poplar'
]
target_variable = 'Yield'
df = df[input_features + [target_variable]].dropna()

# 🔢 Features and Target
X = df[input_features].values
y = df[[target_variable]].values

# 📏 Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 📊 Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 🧠 ANN Model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss=MeanSquaredError())
early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_ann_model.h5", monitor='val_loss', save_best_only=True, verbose=0)

# 🚀 Train ANN
history = model.fit(X_train, y_train, epochs=1500, batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, checkpoint], verbose=1)

# 🔄 Reload best model
model = load_model("best_ann_model.h5", compile=False)
model.compile(optimizer='adam', loss=MeanSquaredError())

# 🎯 Evaluate ANN
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
print(f"\n✅ ANN Test MSE: {mean_squared_error(y_true, y_pred):.4f}")
print(f"✅ ANN R² Score: {r2_score(y_true, y_pred):.4f}")

# 🌲 Random Forest Model
X_rf = df[input_features]
y_rf = df[target_variable]
X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_rf_train, y_rf_train)
y_rf_pred = rf_model.predict(X_rf_test)
print(f"\n✅ RF Test MSE: {mean_squared_error(y_rf_test, y_rf_pred):.4f}")
print(f"✅ RF R² Score: {r2_score(y_rf_test, y_rf_pred):.4f}")

# 📊 ROBUSTNESS TESTS FOR MULTIPLE NOISE LEVELS
noise_levels = [0.01, 0.05, 0.1, 0.2]

print("\n📉 ROBUSTNESS COMPARISON")
for noise in noise_levels:
    # ANN noise
    X_test_noisy = np.clip(X_test + np.random.normal(0, noise, X_test.shape), 0, 1)
    y_pred_noisy_scaled = model.predict(X_test_noisy)
    y_pred_noisy = scaler_y.inverse_transform(y_pred_noisy_scaled)
    mse_ann = mean_squared_error(y_true, y_pred_noisy)
    r2_ann = r2_score(y_true, y_pred_noisy)

    # RF noise
    scaler_rf = MinMaxScaler()
    X_rf_scaled = scaler_rf.fit_transform(X_rf)
    _, X_rf_test_scaled = train_test_split(X_rf_scaled, test_size=0.2, random_state=42)
    X_rf_test_noisy = np.clip(X_rf_test_scaled + np.random.normal(0, noise, X_rf_test_scaled.shape), 0, 1)
    X_rf_test_noisy_orig = pd.DataFrame(scaler_rf.inverse_transform(X_rf_test_noisy), columns=X_rf.columns)
    y_rf_pred_noisy = rf_model.predict(X_rf_test_noisy_orig)
    mse_rf = mean_squared_error(y_rf_test, y_rf_pred_noisy)
    r2_rf = r2_score(y_rf_test, y_rf_pred_noisy)

    # 📌 Print results
    print(f"\n📛 Noise Level: {noise}")
    print(f"   🔹 ANN   -> MSE: {mse_ann:.4f}, R²: {r2_ann:.4f}")
    print(f"   🔸 RF    -> MSE: {mse_rf:.4f}, R²: {r2_rf:.4f}")
