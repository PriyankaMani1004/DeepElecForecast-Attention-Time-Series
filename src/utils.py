import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def inverse_transform(scaler, data):
    
    #Inverse transform scaled data to original scale.    
    #Parameters:
      #scaler: fitted MinMaxScaler
      #data: numpy array (n_samples, steps, 1)    
    #Returns:
      #data_inv: numpy array in original scale
    
    if data.ndim == 3:
        n_samples, steps, n_features = data.shape
        data_flat = data.reshape(-1, n_features)
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(n_samples, steps, n_features)

    elif data.ndim == 2:
        n_samples, steps = data.shape
        data_flat = data.reshape(-1, 1)
        data_inv = scaler.inverse_transform(data_flat)
        return data_inv.reshape(n_samples, steps)

    else:
        raise ValueError("Data must be 2D or 3D for inverse transform")
    return data_inv

def compute_metrics(y_true, y_pred):
    
    #Compute RMSE and MAE for multi-step forecasting.    
    #Parameters:y_true, y_pred: numpy arrays of shape (samples, steps, 1)    
    #Returns: - rmse, mae: float
    
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    return rmse, mae

def plot_predictions(y_true, y_pred, title="Forecast vs Actual", steps_to_plot=100):
    
    #Plot predictions vs actual for multi-step forecasts.    
    #Parameters:
      #y_true, y_pred: numpy arrays (samples, steps, 1)
      #steps_to_plot: int, number of timesteps to visualize

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    plt.figure(figsize=(12,6))
    plt.plot(y_true_flat[:steps_to_plot], label="Actual")
    plt.plot(y_pred_flat[:steps_to_plot], label="Predicted")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Electricity Consumption (kW)")
    plt.legend()
    plt.show()

def plot_attention_weights(attention_weights, past_steps, title="Attention Heatmap"):
    
    #Plot attention weights as a heatmap.    
    #Parameters:
      #attention_weights: numpy array of shape (samples, past_steps)
      #past_steps: int, length of input sequence
    
    import seaborn as sns
    
    # Average over all samples
    avg_weights = np.mean(attention_weights, axis=0)
    
    plt.figure(figsize=(10,4))
    sns.heatmap(avg_weights.reshape(1,-1), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Past Timesteps")
    plt.ylabel("Attention")
    plt.show()

