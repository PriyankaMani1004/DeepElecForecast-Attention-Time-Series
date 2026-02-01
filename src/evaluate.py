import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from src.data_loader import load_processed_data
from src.utils import inverse_transform, compute_metrics, plot_predictions, plot_attention_weights

#Paths
processed_folder = "data/processed/"
results_folder = "results"
models_folder = os.path.join(results_folder, "models")
plots_folder = os.path.join(results_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

#Load processed data
X_train, y_train, X_test, y_test, scaler = load_processed_data(processed_folder)

#Evaluate Baseline LSTM
baseline_path = os.path.join(models_folder, "lstm_baseline.keras")
baseline_model = load_model(baseline_path)

# Predict
y_pred_baseline = baseline_model.predict(X_test)

# Inverse scale
y_test_inv = inverse_transform(scaler, y_test)
y_pred_baseline_inv = inverse_transform(scaler, y_pred_baseline)

# Compute metrics
rmse_base, mae_base = compute_metrics(y_test_inv, y_pred_baseline_inv)
print(f"Baseline LSTM -> RMSE: {rmse_base:.3f}, MAE: {mae_base:.3f}")

# Plot predictions
plot_predictions(y_test_inv, y_pred_baseline_inv, title="Baseline LSTM Forecast", steps_to_plot=200)

#Evaluate Attention LSTM
attn_path = os.path.join(models_folder, "lstm_attention.keras")
attn_model = load_model(attn_path)

# Predict
y_pred_attn = attn_model.predict(X_test)
y_pred_attn_inv = inverse_transform(scaler, y_pred_attn)

# Compute metrics
rmse_attn, mae_attn = compute_metrics(y_test_inv, y_pred_attn_inv)
print(f"Attention LSTM -> RMSE: {rmse_attn:.3f}, MAE: {mae_attn:.3f}")

# Plot predictions
plot_predictions(y_test_inv, y_pred_attn_inv, title="Attention LSTM Forecast", steps_to_plot=200)