from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_baseline(input_shape, output_steps, units=64, dropout=0.2, learning_rate=0.001):
    
    #Build a simple LSTM model for multi-step forecasting.
    
    #Parameters:
      #input_shape: tuple, (past_steps, n_features)
      #output_steps: int, number of future steps to predict
      #units: int, LSTM units
      #dropout: float, dropout rate
      #learning_rate: float, optimizer learning rate
    
    #Returns:model: compiled Keras model
    
    
    model = Sequential()    
    # LSTM layer
    model.add(LSTM(units, activation='tanh', input_shape=input_shape))    
    # Optional dropout
    model.add(Dropout(dropout))    
    # Dense output layer for multi-step forecasting
    model.add(Dense(output_steps))
    # Compile model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model