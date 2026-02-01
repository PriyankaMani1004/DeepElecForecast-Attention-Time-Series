import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

from src.models.custom_layers import Attention

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "results/models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Model builders
def build_lstm_baseline(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64)(inputs)
    outputs = Dense(output_steps)(x)
    return Model(inputs, outputs)

def build_lstm_attention(input_shape, output_steps):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Attention()(x)
    outputs = Dense(output_steps)(x)
    return Model(inputs, outputs)