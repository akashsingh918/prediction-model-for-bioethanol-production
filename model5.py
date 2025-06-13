# Model 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from google.colab import files
uploaded = files.upload()

# Load cleaned dataset
df = pd.read_csv("glucose_yield_data_cleaned.csv")

# 🔧 Feature Engineering
df['Severity'] = df['CiA'] * df['T_C'] * df['T_min']
df['log_Severity'] = np.log1p(df['Severity'])
df['T_C_x_CiA'] = df['T_C'] * df['CiA']
df['T_min_sq'] = df['T_min'] ** 2
df['kiA_x_E'] = df['kiA'] * df['E_FPU_g']

# 📊 Top Features by Permutation Importance
features = ['T_C_x_CiA', 'T_C', 'kiA_x_E', 'kiA', 'T_min_sq']
X = df[features].values
y = df['CGlc_mean'].values.ravel()

# Scale features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train_combined = np.concatenate([X_train, X_val])
y_train_combined = np.concatenate([y_train, y_val])

# 🌲 Retrained Random Forest Model on Top Features
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train_combined, y_train_combined)
y_pred_rf = rf.predict(X_test)

# 📊 Metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"\n🌲 Final Random Forest (Top Features) MSE: {mse_rf:.4f}")
print(f"📈 Final Random Forest (Top Features) R² Score: {r2_rf:.4f}")

# 📈 Plot: True vs Predicted
y_min, y_max = min(y_test), max(y_test)
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_min, y_max], [y_min, y_max], 'r--')
plt.xlabel("True Glucose Yield")
plt.ylabel("Predicted Glucose Yield")
plt.title("RF (Top Features): True vs Predicted Glucose Yield")
plt.grid(True)
plt.tight_layout()
plt.show()

# 💾 Save the model and scaler
joblib.dump(rf, "rf_model_top_features.pkl")
joblib.dump(scaler_X, "scaler_top_features.pkl")

# 📤 Export predictions
pd.DataFrame({
    "True_CGlc_mean": y_test,
    "Predicted_CGlc_mean": y_pred_rf
}).to_csv("rf_top_features_predictions.csv", index=False)

# 🧪 Noise Robustness Test
def add_noise(X, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)
    return np.clip(X + noise, 0, 1)

X_noisy = add_noise(X_scaled, noise_level=0.05)
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_noisy, y, test_size=0.2, random_state=42)
rf.fit(X_train_n, y_train_n)
y_pred_n = rf.predict(X_test_n)

mse_noise = mean_squared_error(y_test_n, y_pred_n)
r2_noise = r2_score(y_test_n, y_pred_n)

print("\n🧪 Noise Robustness Test (Top Features):")
print(f"MSE with Noise: {mse_noise:.4f}")
print(f"R² with Noise: {r2_noise:.4f}")
