# ✅ MULTIMODEL COMPARISON: Top 3, Top 4, and All Features
# https://doi.org/10.1002/jctb.5456

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from google.colab import files

# ====== Upload Files ======
uploaded = files.upload()
file_names = list(uploaded.keys())
df_train1 = pd.read_csv([f for f in file_names if "table3" in f][0])
df_train2 = pd.read_csv([f for f in file_names if "table5" in f][0])
df_test = pd.read_csv([f for f in file_names if "table4" in f][0])

# ====== Combine & Clean ======
df_train = pd.concat([df_train1, df_train2], ignore_index=True)
rename_cols = {
    "time(min)": "pretreatment_time_min",
    "S(%)": "biomass_concentration_pct",
    "Leh(FPU/gwis)": "enzyme_loading_fpugwis",
    "Seh (%w/v)": "substrate_loading_wv_pct",
    "Deh (%5)": "cellulose_digestibility_pct",
    "YGLC (%)": "glucose_yield_pct"
}
df_train.rename(columns=rename_cols, inplace=True)
df_test.rename(columns=rename_cols, inplace=True)

all_features = ['pretreatment_time_min', 'biomass_concentration_pct', 'enzyme_loading_fpugwis',
                'substrate_loading_wv_pct', 'cellulose_digestibility_pct']
target = 'glucose_yield_pct'

X_train_full = df_train[all_features]
y_train = df_train[target]
X_test_full = df_test[all_features]
y_test = df_test[target]

# ====== Feature Importance ======
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_full, y_train)
perm = permutation_importance(rf, X_train_full, y_train, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': perm.importances_mean
}).sort_values(by='Importance', ascending=False)

# ====== Define Feature Sets ======
feature_sets = {
    'Top 3': importance_df['Feature'].head(3).tolist(),
    'Top 4': importance_df['Feature'].head(4).tolist(),
    'All': all_features
}

# ====== Model Training Function ======
def run_ann(selected_features):
    # Scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_full[selected_features])
    X_test_scaled = scaler_X.transform(X_test_full[selected_features])
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Train/Val split
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train_scaled, test_size=0.2, random_state=42)

    # Model
    model = Sequential([
        Input(shape=(X_train_final.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

    model.fit(X_train_final, y_train_final,
              validation_data=(X_val, y_val),
              epochs=1000, batch_size=16,
              callbacks=[early_stop], verbose=0)

    # Evaluation on clean test
    pred_scaled = model.predict(X_test_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    true = scaler_y.inverse_transform(y_test_scaled)
    mse_clean = mean_squared_error(true, pred)
    r2_clean = r2_score(true, pred)

    # Noise robustness test
    noise = np.random.normal(0, 0.05, X_test_scaled.shape)
    X_test_noisy = np.clip(X_test_scaled + noise, 0, 1)
    pred_noisy_scaled = model.predict(X_test_noisy)
    pred_noisy = scaler_y.inverse_transform(pred_noisy_scaled)
    mse_noisy = mean_squared_error(true, pred_noisy)
    r2_noisy = r2_score(true, pred_noisy)

    return round(mse_clean, 4), round(r2_clean, 4), round(mse_noisy, 4), round(r2_noisy, 4)

# ====== Run Models and Report ======
print("\n📊 COMPARATIVE RESULTS:")
print("-" * 45)
for label, feats in feature_sets.items():
    mse_c, r2_c, mse_n, r2_n = run_ann(feats)
    print(f"{label} Features:")
    print(f"  ✅ Clean Data  → MSE = {mse_c}, R² = {r2_c}")
    print(f"  🧪 5% Noisy     → MSE = {mse_n}, R² = {r2_n}")
    print("-" * 45)
