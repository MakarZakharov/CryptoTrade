import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31')
data = df[['Close']].values

# 2. Масштабування
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Підготовка вибірки
X = []
y = []
window = 60  # 60 днів історії

for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# 4. Побудова моделі
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile
