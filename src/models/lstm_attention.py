from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Permute, RepeatVector, Activation, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def build_lstm_attention(input_shape, output_steps, units=64, dropout=0.2, learning_rate=0.001):
    
    #Build an LSTM model with attention mechanism for multi-step forecasting.
    
    #Parameters:
     #input_shape: tuple, (past_steps, n_features)
     #output_steps: int, number of future steps to predict
     #units: int, LSTM units
     #dropout: float, dropout rate
     #learning_rate: float, optimizer learning rate
    
    #Returns:model: compiled Keras Model
    
    
    # Input layer
    inputs = Input(shape=input_shape)  # (past_steps, features)
    
    # LSTM layer
    lstm_out = LSTM(units, activation='tanh', return_sequences=True)(inputs)
    
    # Attention mechanism
    # Step 1: compute attention scores
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Lambda(lambda x: K.squeeze(x, axis=-1))(attention)  # shape: (batch, past_steps)
    attention = Activation('softmax', name='attention_weights')(attention)  # attention weights
    
    # Step 2: compute context vector
    attention = RepeatVector(units)(attention)              # shape: (batch, units, past_steps)
    attention = Permute([2,1])(attention)                  # shape: (batch, past_steps, units)
    context = Multiply()([lstm_out, attention])           # shape: (batch, past_steps, units)
    context = Lambda(lambda x: K.sum(x, axis=1))(context) # sum over timesteps (batch, units)
    
    # Dropout
    context = Dropout(dropout)(context)
    
    # Dense output
    outputs = Dense(output_steps)(context)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    return model
