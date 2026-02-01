import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_raw_data(path="data/raw/electricity.csv"):   
    #Load raw electricity CSV and return a DataFrame with datetime index
    
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def preprocess_series(series, resample_freq="H"):
   
    #Preprocess a single pandas Series:
       #Fill missing values
      # Resample to desired frequency (e.g., hourly)    
    #Returns a cleaned series
    
    series = series.fillna(method='ffill')
    series = series.resample(resample_freq).mean()
    return series

def scale_series(series):    
    #Scale series using MinMaxScaler to [0,1]
    #Returns scaled series and scaler object    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    return scaled, scaler

def create_sequences(data, past_steps=48, future_steps=12):    
    #Create input-output sequences for LSTM
      #data: scaled numpy array (n_samples, 1)
    #Returns:
      #X: input sequences
      #y: output sequences
    
    X, y = [], []
    for i in range(len(data) - past_steps - future_steps + 1):
        X.append(data[i:i+past_steps])
        y.append(data[i+past_steps:i+past_steps+future_steps])
    return np.array(X), np.array(y)

def preprocess_and_save(client="MT_001",
                        raw_path="data/raw/electricity.csv",
                        processed_folder="data/processed/",
                        resample_freq="H",
                        past_steps=48,
                        future_steps=12,
                        train_ratio=0.8):
  
    
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Load raw data
    df = load_raw_data(raw_path)
    
    # Select client series
    series = df[client]
    
    #Preprocess and resample
    series = preprocess_series(series, resample_freq=resample_freq)
    
    #Scale
    scaled, scaler = scale_series(series)
    
    #Create sequences
    X, y = create_sequences(scaled, past_steps=past_steps, future_steps=future_steps)
    
    #Train/test split
    split_idx = int(len(X) * train_ratio)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    #Save processed arrays and scaler
    np.save(os.path.join(processed_folder, "X_train.npy"), X_train)
    np.save(os.path.join(processed_folder, "y_train.npy"), y_train)
    np.save(os.path.join(processed_folder, "X_test.npy"), X_test)
    np.save(os.path.join(processed_folder, "y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(processed_folder, "scaler.pkl"))
    
    print(f"Preprocessing complete. Processed data saved in '{processed_folder}'")
    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    
    return X_train, y_train, X_test, y_test, scaler