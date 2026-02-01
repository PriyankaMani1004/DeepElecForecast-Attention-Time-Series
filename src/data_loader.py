import pandas as pd
import numpy as np
import joblib
import os

def load_raw_data(path="data/raw/electricity.csv"):
   
    #Load the raw electricity CSV.    
    #Parameters:- path: str, path to CSV file    
    #Returns:- df: pandas DataFrame with DateTime index
    
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def load_processed_data(processed_folder="data/processed/"):
    
    #Load preprocessed and saved data arrays (X_train, y_train, X_test, y_test) and the fitted scaler.    
    #Parameters: processed_folder: str, path to processed folder    
    #Returns: X_train, y_train, X_test, y_test, scaler
    
    X_train = np.load(os.path.join(processed_folder, "X_train.npy"))
    y_train = np.load(os.path.join(processed_folder, "y_train.npy"))
    X_test  = np.load(os.path.join(processed_folder, "X_test.npy"))
    y_test  = np.load(os.path.join(processed_folder, "y_test.npy"))
    scaler  = joblib.load(os.path.join(processed_folder, "scaler.pkl"))
    
    return X_train, y_train, X_test, y_test, scaler

def train_test_split_series(series, train_size=0.8):
    
   # Split a pandas series into train and test sets.    
    #Parameters: series: pd.Series, train_size: float, proportion of data for training    
    #Returns:train: pd.Series, test: pd.Series
    
    n = len(series)
    split = int(n * train_size)
    train = series[:split]
    test = series[split:]
    return train, test