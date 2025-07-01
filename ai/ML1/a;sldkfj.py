import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

class LSTMSequencePredictor:
    """
    LSTM Neural Network for Cryptocurrency Price Sequence Prediction
    """
    
    def __init__(self, sequence_length=60, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def load_and_prepare_data(self, csv_path=None):
        """
        Загружает и подготавливает данные для обучения
        """
        if csv_path is None:
            # Путь к данным BTC/USDT
            csv_path = os.path.join("..", "..", "data", "binance", "BTCUSDT", "1d", "2018_01_01-now.csv")
        
        # Загрузка данных
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Используем только цену закрытия
        data = df[['close']].values
        
        # Нормализация данных
        scaled_data = self.scaler.fit_transform(data)
        
        return scaled_data
    
    def create_sequences(self, data, train_size=0.8):
        """
        Создает последовательности для обучения LSTM
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        X, y = np.array(X), np.array(y)
        
        # Разделение на обучающую и тестовую выборки
        split_idx = int(len(X) * train_size)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, lstm_units=[50, 50], dropout_rate=0.2):
        """
        Создает архитектуру LSTM модели
        """
        self.model = Sequential()
        
        # Первый LSTM слой
        self.model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(dropout_rate))
        
        # Дополнительные LSTM слои
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(dropout_rate))
        
        # Выходной слой
        self.model.add(Dense(units=1))
        
        # Компиляция модели
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Обучает LSTM модель
        """
        if self.model is None:
            self.build_model()
        
        # Колбэки для улучшения обучения
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Валидационные данные
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Обучение
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Делает предсказания
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Вызовите train() сначала.")
        
        predictions = self.model.predict(X)
        # Обратное масштабирование
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Оценивает качество модели
        """
        # Предсказания
        predictions = self.model.predict(X_test)
        
        # Обратное масштабирование
        y_test_scaled = self.scaler.inverse_transform(y_test)
        predictions_scaled = self.scaler.inverse_transform(predictions)
        
        # Метрики
        mse = mean_squared_error(y_test_scaled, predictions_scaled)
        mae = mean_absolute_error(y_test_scaled, predictions_scaled)
        rmse = np.sqrt(mse)
        
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        return {
            'mse': mse,
            'mae': mae, 
            'rmse': rmse,
            'predictions': predictions_scaled,
            'actual': y_test_scaled
        }
    
    def plot_training_history(self):
        """
        Визуализирует историю обучения
        """
        if self.history is None:
            print("Модель не обучена.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, y_actual, y_pred, start_idx=0, end_idx=200):
        """
        Визуализирует предсказания vs реальные значения
        """
        plt.figure(figsize=(15, 6))
        
        x_range = range(start_idx, min(end_idx, len(y_actual)))
        
        plt.plot(x_range, y_actual[start_idx:end_idx], 
                label='Actual Price', linewidth=2, alpha=0.8)
        plt.plot(x_range, y_pred[start_idx:end_idx], 
                label='Predicted Price', linewidth=2, alpha=0.8)
        
        plt.title('BTC Price Prediction vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    """
    Основная функция для демонстрации работы LSTM модели
    """
    print("🚀 Запуск LSTM модели для предсказания цены BTC...")
    
    # Создание экземпляра модели
    lstm_predictor = LSTMSequencePredictor(sequence_length=60)
    
    try:
        # Загрузка и подготовка данных
        print("📊 Загрузка данных...")
        scaled_data = lstm_predictor.load_and_prepare_data()
        
        # Создание последовательностей
        print("🔄 Создание последовательностей...")
        X_train, X_test, y_train, y_test = lstm_predictor.create_sequences(scaled_data)
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        
        # Построение модели
        print("🏗️ Построение LSTM модели...")
        lstm_predictor.build_model(lstm_units=[50, 50, 25])
        print(lstm_predictor.model.summary())
        
        # Обучение
        print("🎯 Обучение модели...")
        lstm_predictor.train(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            epochs=50,
            batch_size=32
        )
        
        # Оценка качества
        print("📈 Оценка модели...")
        results = lstm_predictor.evaluate(X_test, y_test)
        
        # Визуализация
        print("📊 Построение графиков...")
        lstm_predictor.plot_training_history()
        lstm_predictor.plot_predictions(
            results['actual'], 
            results['predictions'],
            end_idx=100
        )
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Убедитесь, что файл с данными существует по указанному пути.")

if __name__ == "__main__":
    main()