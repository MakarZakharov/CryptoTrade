import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CryptoPricePredictor:
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.model = None
        
    def get_binance_data(self, symbol, interval, start_str, end_str=None):
        """Отримання даних з Binance API"""
        url = "https://api.binance.com/api/v3/klines"
        start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else None

        all_klines = []
        limit = 1000
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ts,
                "limit": limit
            }
            if end_ts:
                params["endTime"] = end_ts

            try:
                response = requests.get(url, params=params)
                data = response.json()
                if not data or isinstance(data, dict) and "code" in data:
                    break

                all_klines += data
                start_ts = data[-1][6] + 1

                if len(data) < limit:
                    break
            except Exception as e:
                print(f"Помилка при отриманні даних: {e}")
                break
                
        return all_klines

    def process_binance_data(self, klines):
        """Обробка даних з Binance"""
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.astype({
            "open": "float",
            "high": "float", 
            "low": "float",
            "close": "float",
            "volume": "float"
        })
        
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df.set_index("timestamp", inplace=True)
        return df

    def prepare_data(self, data):
        """Підготовка даних для навчання"""
        # Використовуємо ціну закриття
        close_prices = data[['close']].values
        
        # Масштабування
        scaled_data = self.scaler.fit_transform(close_prices)
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i-self.window_size:i, 0])
            y.append(scaled_data[i, 0])
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Побудова LSTM моделі"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def train_model(self, symbol="BTCUSDT", start_date="2020-01-01", use_cross_validation=True):
        """Навчання моделі з cross-validation для запобігання перенавчанню"""
        print(f"Завантаження даних для {symbol}...")
        
        # Отримання даних
        klines = self.get_binance_data(symbol, "1d", start_date)
        if not klines:
            print("Не вдалося отримати дані!")
            return False
            
        df = self.process_binance_data(klines)
        print(f"Завантажено {len(df)} записів")
        
        # Підготовка даних
        X, y = self.prepare_data(df)
        
        if use_cross_validation:
            # Використання TimeSeriesSplit для часових рядів
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            best_model = None
            best_score = float('inf')
            
            print("Використання 5-fold cross-validation для часових рядів...")
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"Навчання на fold {fold + 1}/5...")
                
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Зміна форми для LSTM
                X_train_cv = X_train_cv.reshape((X_train_cv.shape[0], X_train_cv.shape[1], 1))
                X_val_cv = X_val_cv.reshape((X_val_cv.shape[0], X_val_cv.shape[1], 1))
                
                # Побудова моделі для кожного fold
                model = self.build_model((X_train_cv.shape[1], 1))
                
                # Додавання callbacks для запобігання перенавчанню
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
                
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=0.0001,
                    verbose=0
                )
                
                # Навчання моделі
                model.fit(
                    X_train_cv, y_train_cv,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val_cv, y_val_cv),
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                # Оцінка на валідаційній вибірці
                val_pred = model.predict(X_val_cv, verbose=0)
                val_pred_scaled = self.scaler.inverse_transform(val_pred)
                y_val_scaled = self.scaler.inverse_transform(y_val_cv.reshape(-1, 1))
                
                fold_rmse = np.sqrt(mean_squared_error(y_val_scaled, val_pred_scaled))
                cv_scores.append(fold_rmse)
                
                # Зберігаємо найкращу модель
                if fold_rmse < best_score:
                    best_score = fold_rmse
                    best_model = model
                    
            print(f"CV RMSE: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")
            self.model = best_model
            self.cv_scores = cv_scores
            
        # Фінальне навчання на всіх даних для остаточної оцінки
        train_size = int(len(X) * 0.7)  # 70% для навчання
        val_size = int(len(X) * 0.15)   # 15% для валідації
        # 15% для тесту
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        # Зміна форми для LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Фінальне навчання найкращої моделі
        print("Фінальне навчання моделі...")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=0.0001,
            verbose=1
        )
        
        if not use_cross_validation:
            self.model = self.build_model((X_train.shape[1], 1))
            
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Оцінка моделі на всіх наборах
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        # Зворотне масштабування
        train_pred = self.scaler.inverse_transform(train_pred)
        val_pred = self.scaler.inverse_transform(val_pred)
        test_pred = self.scaler.inverse_transform(test_pred)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_val_actual = self.scaler.inverse_transform(y_val.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Розрахунок метрик
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_actual, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))
        
        train_mae = mean_absolute_error(y_train_actual, train_pred)
        val_mae = mean_absolute_error(y_val_actual, val_pred)
        test_mae = mean_absolute_error(y_test_actual, test_pred)
        
        train_r2 = r2_score(y_train_actual, train_pred)
        val_r2 = r2_score(y_val_actual, val_pred)
        test_r2 = r2_score(y_test_actual, test_pred)
        
        print(f"\n=== РЕЗУЛЬТАТИ НАВЧАННЯ ===")
        print(f"Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R²: {train_r2:.4f}")
        print(f"Val RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R²: {test_r2:.4f}")
        
        # Збереження даних для візуалізації
        self.df = df
        self.train_size = train_size
        self.val_size = val_size
        self.train_pred = train_pred
        self.val_pred = val_pred
        self.test_pred = test_pred
        self.y_train_actual = y_train_actual
        self.y_val_actual = y_val_actual
        self.y_test_actual = y_test_actual
        self.history = history
        self.metrics = {
            'train_rmse': train_rmse, 'val_rmse': val_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'val_mae': val_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'val_r2': val_r2, 'test_r2': test_r2
        }
        
        return True

    def predict_future(self, days=30):
        """Прогнозування на майбутнє"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        # Останні дані для прогнозу
        last_data = self.df['close'].values[-self.window_size:]
        last_scaled = self.scaler.transform(last_data.reshape(-1, 1))
        
        predictions = []
        current_batch = last_scaled.reshape((1, self.window_size, 1))
        
        for i in range(days):
            pred = self.model.predict(current_batch)[0]
            predictions.append(pred[0])
            
            # Оновлення batch для наступного прогнозу
            current_batch = np.append(current_batch[:, 1:, :], 
                                    pred.reshape(1, 1, 1), axis=1)
        
        # Зворотне масштабування
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Створення дат для прогнозу
        last_date = self.df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions.flatten()
        })

    def calculate_technical_indicators(self, prices, window=14):
        """Розрахунок технічних індикаторів"""
        if len(prices) < window:
            return None, None, None, None
            
        # RSI (Relative Strength Index)
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-window:])
        avg_loss = np.mean(loss[-window:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Moving Averages
        sma_short = np.mean(prices[-5:])  # 5-денна
        sma_long = np.mean(prices[-window:])  # 14-денна
        
        # Волатильність
        volatility = np.std(prices[-window:]) / np.mean(prices[-window:]) * 100
        
        return rsi, sma_short, sma_long, volatility

    def simulate_trading(self, initial_balance=100.0, strategy='enhanced'):
        """Покращена симуляція торгівлі з технічними індикаторами"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        # Використовуємо тестові дані для симуляції
        test_start_idx = self.window_size + self.train_size
        test_prices = self.df['close'].iloc[test_start_idx:test_start_idx + len(self.test_pred)]
        test_predictions = self.test_pred.flatten()
        
        balance = initial_balance
        position = 0  # кількість монет
        trades = []
        balances = [initial_balance]
        
        # Отримуємо історичні дані для технічних індикаторів
        historical_prices = self.df['close'].iloc[:test_start_idx + len(test_predictions)].values
        
        for i in range(len(test_predictions) - 1):
            current_price = test_prices.iloc[i]
            next_predicted = test_predictions[i + 1]
            current_predicted = test_predictions[i] if i > 0 else current_price
            
            # Отримуємо дані до поточного моменту для індикаторів
            price_history = historical_prices[:test_start_idx + i + 1]
            
            if strategy == 'enhanced' and len(price_history) >= 14:
                # Розрахунок технічних індикаторів
                rsi, sma_short, sma_long, volatility = self.calculate_technical_indicators(price_history)
                
                # Покращена логіка торгівлі
                prediction_change = (next_predicted - current_price) / current_price * 100
                ma_signal = sma_short > sma_long  # True = bullish, False = bearish
                
                # Сигнали для покупки
                buy_signals = 0
                if prediction_change > 0.2:  # прогноз зростання > 0.2%
                    buy_signals += 1
                if rsi < 30:  # oversold
                    buy_signals += 1
                if ma_signal and volatility < 5:  # висхідний тренд, низька волатильність
                    buy_signals += 1
                if next_predicted > current_price * 1.003:  # прогноз зростання > 0.3%
                    buy_signals += 1
                
                # Сигнали для продажу
                sell_signals = 0
                if prediction_change < -0.2:  # прогноз падіння > 0.2%
                    sell_signals += 1
                if rsi > 70:  # overbought
                    sell_signals += 1
                if not ma_signal:  # спадний тренд
                    sell_signals += 1
                if next_predicted < current_price * 0.997:  # прогноз падіння > 0.3%
                    sell_signals += 1
                
                # Ризик-менеджмент: обмежуємо розмір позиції
                max_position_value = initial_balance * 0.8  # максимум 80% балансу в позиції
                
                # Логіка торгівлі з множинними сигналами
                if buy_signals >= 2 and position == 0 and balance > current_price:
                    # Купуємо частину балансу (50-80% залежно від сигналів)
                    investment_ratio = min(0.5 + (buy_signals - 2) * 0.1, 0.8)
                    investment_amount = balance * investment_ratio
                    coins_to_buy = investment_amount / current_price
                    
                    position += coins_to_buy
                    balance -= investment_amount
                    trades.append(('BUY', current_price, coins_to_buy, test_prices.index[i]))
                    
                elif sell_signals >= 2 and position > 0:
                    # Продаємо частину або всю позицію
                    sell_ratio = min(0.5 + (sell_signals - 2) * 0.2, 1.0)
                    coins_to_sell = position * sell_ratio
                    
                    sell_amount = coins_to_sell * current_price
                    balance += sell_amount
                    position -= coins_to_sell
                    trades.append(('SELL', current_price, coins_to_sell, test_prices.index[i]))
                    
                # Stop-loss: продаємо якщо втрати > 5%
                elif position > 0:
                    avg_buy_price = (initial_balance - balance + position * current_price) / position if position > 0 else current_price
                    if current_price < avg_buy_price * 0.95:  # втрати > 5%
                        sell_amount = position * current_price
                        balance += sell_amount
                        trades.append(('SELL', current_price, position, test_prices.index[i]))
                        position = 0
                        
            else:
                # Проста стратегія з дуже низькими порогами
                prediction_change = (next_predicted - current_price) / current_price * 100
                
                if prediction_change > 0.05 and position == 0 and balance > current_price:  # купуємо при зростанні > 0.05%
                    coins_to_buy = balance * 0.5 / current_price  # інвестуємо 50% балансу
                    position += coins_to_buy
                    balance -= coins_to_buy * current_price
                    trades.append(('BUY', current_price, coins_to_buy, test_prices.index[i]))
                    
                elif prediction_change < -0.05 and position > 0:  # продаємо при падінні > 0.05%
                    sell_amount = position * current_price
                    balance += sell_amount
                    trades.append(('SELL', current_price, position, test_prices.index[i]))
                    position = 0
            
            # Розрахунок поточного балансу
            current_balance = balance + (position * current_price)
            balances.append(current_balance)
        
        # Фінальний продаж якщо залишились монети
        if position > 0:
            final_price = test_prices.iloc[-1]
            balance += position * final_price
            trades.append(('SELL', final_price, position, test_prices.index[-1]))
            position = 0
        
        final_balance = balance
        
        # Розрахунок додаткових метрик
        if trades:
            buy_trades = [t for t in trades if t[0] == 'BUY']
            sell_trades = [t for t in trades if t[0] == 'SELL']
            
            total_fees = len(trades) * 0.1  # 0.1% комісія за торгівлю
            final_balance -= total_fees
            
            win_trades = 0
            total_trades_pairs = min(len(buy_trades), len(sell_trades))
            
            if total_trades_pairs > 0:
                for i in range(total_trades_pairs):
                    buy_price = buy_trades[i][1]
                    sell_price = sell_trades[i][1]
                    if sell_price > buy_price:
                        win_trades += 1
                        
                win_rate = (win_trades / total_trades_pairs) * 100
            else:
                win_rate = 0
        else:
            total_fees = 0
            win_rate = 0
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit_loss': final_balance - initial_balance,
            'profit_percentage': ((final_balance - initial_balance) / initial_balance) * 100,
            'trades': trades,
            'balance_history': balances,
            'test_dates': test_prices.index,
            'total_fees': total_fees,
            'win_rate': win_rate,
            'strategy': strategy
        }

    def plot_results(self, symbol="BTCUSDT"):
        """Візуалізація результатів з покращеною видимістю початку"""
        if self.model is None:
            print("Модель не навчена!")
            return
            
        # Симуляція торгівлі
        trading_results = self.simulate_trading_all_periods(initial_balance=100.0)
            
        plt.figure(figsize=(18, 15))
        
        # 1. Історія навчання
        plt.subplot(3, 2, 1)
        plt.plot(self.history.history['loss'], label='Втрати навчання', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Втрати валідації', linewidth=2)
        plt.title('Історія навчання моделі', fontsize=14, fontweight='bold')
        plt.xlabel('Епоха')
        plt.ylabel('Втрати')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Порівняння прогнозів з реальними цінами (з покращеною видимістю початку)
        plt.subplot(3, 2, 2)
        
        # Підготовка даних для графіку
        train_dates = self.df.index[self.window_size:self.window_size+len(self.train_pred)]
        test_dates = self.df.index[self.window_size+len(self.train_pred):self.window_size+len(self.train_pred)+len(self.test_pred)]
        
        # Показуємо всі дані але з акцентом на початок
        plt.plot(self.df.index, self.df['close'], label='Реальна ціна', alpha=0.6, linewidth=1)
        plt.plot(train_dates, self.train_pred, label='Прогноз (навчання)', alpha=0.8, linewidth=2)
        plt.plot(test_dates, self.test_pred, label='Прогноз (тест)', alpha=0.9, linewidth=2)
        
        # Виділяємо початок даних
        early_data = self.df.head(200)  # перші 200 днів
        plt.axvspan(early_data.index[0], early_data.index[-1], alpha=0.1, color='green', label='Початковий період')
        
        plt.title(f'Прогнозування ціни {symbol}', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Ціна (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Крупний план початку даних
        plt.subplot(3, 2, 3)
        start_period = min(300, len(self.df))  # перші 300 днів або менше
        early_df = self.df.head(start_period)
        plt.plot(early_df.index, early_df['close'], label='Реальна ціна (початок)', linewidth=3, color='blue')
        plt.title('Детальний вигляд початкового періоду', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Ціна (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 4. Прогноз на майбутнє
        plt.subplot(3, 2, 4)
        future_pred = self.predict_future(30)
        
        # Останні 100 днів + прогноз
        recent_data = self.df.tail(100)
        plt.plot(recent_data.index, recent_data['close'], label='Історичні дані', linewidth=3, color='darkblue')
        plt.plot(future_pred['date'], future_pred['predicted_price'], 
                label='Прогноз на 30 днів', linewidth=3, linestyle='--', color='red')
        
        plt.title('Прогноз на майбутнє', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Ціна (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 5. Розподіл помилок
        plt.subplot(3, 2, 5)
        test_errors = self.y_test_actual.flatten() - self.test_pred.flatten()
        plt.hist(test_errors, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
        plt.title('Розподіл помилок прогнозування', fontsize=14, fontweight='bold')
        plt.xlabel('Помилка (USDT)')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        
        # 6. Симуляція торгівлі з початковим балансом $100
        plt.subplot(3, 2, 6)
        if trading_results:
            dates_key = 'all_dates' if 'all_dates' in trading_results else 'test_dates'
            plt.plot(trading_results[dates_key], trading_results['balance_history'], 
                    label=f'Баланс (початковий: ${trading_results["initial_balance"]:.0f})', 
                    linewidth=3, color='green')
            plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Початковий баланс')
            
            # Позначаємо угоди
            for trade in trading_results['trades'][:50]:  # показуємо перші 50 угод для читаності
                trade_type, price, amount, date = trade[:4]  # беремо перші 4 елементи
                color = 'green' if trade_type == 'BUY' else 'red'
                marker = '^' if trade_type == 'BUY' else 'v'
                try:
                    balance_idx = list(trading_results[dates_key]).index(date) if date in trading_results[dates_key] else -1
                    if 0 <= balance_idx < len(trading_results['balance_history']):
                        plt.scatter(date, trading_results['balance_history'][balance_idx], 
                                  color=color, marker=marker, s=50, alpha=0.8)
                except:
                    pass  # пропускаємо помилки індексування
        
        plt.title('Симуляція торгівлі ($100 початковий баланс)', fontsize=14, fontweight='bold')
        plt.xlabel('Дата')
        plt.ylabel('Баланс ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Виведення статистики прогнозу
        future_pred = self.predict_future(30)
        print("\n=== СТАТИСТИКА ПРОГНОЗУ ===")
        print(f"Поточна ціна: ${self.df['close'].iloc[-1]:.2f}")
        print(f"Прогноз на завтра: ${future_pred['predicted_price'].iloc[0]:.2f}")
        print(f"Прогноз на тиждень: ${future_pred['predicted_price'].iloc[6]:.2f}")
        print(f"Прогноз на місяць: ${future_pred['predicted_price'].iloc[-1]:.2f}")
        
        change_tomorrow = ((future_pred['predicted_price'].iloc[0] - self.df['close'].iloc[-1]) / self.df['close'].iloc[-1]) * 100
        change_week = ((future_pred['predicted_price'].iloc[6] - self.df['close'].iloc[-1]) / self.df['close'].iloc[-1]) * 100
        change_month = ((future_pred['predicted_price'].iloc[-1] - self.df['close'].iloc[-1]) / self.df['close'].iloc[-1]) * 100
        
        print(f"Зміна на завтра: {change_tomorrow:+.2f}%")
        print(f"Зміна на тиждень: {change_week:+.2f}%")
        print(f"Зміна на місяць: {change_month:+.2f}%")
        
        # Статистика торгівлі
        if trading_results:
            print(f"\n=== РЕЗУЛЬТАТИ ТОРГІВЛІ ({trading_results.get('strategy', 'simple').upper()}) ===")
            print(f"Початковий баланс: ${trading_results['initial_balance']:.2f}")
            print(f"Фінальний баланс: ${trading_results['final_balance']:.2f}")
            print(f"Прибуток/Збиток: ${trading_results['profit_loss']:+.2f}")
            print(f"Відсоток прибутку: {trading_results['profit_percentage']:+.2f}%")
            print(f"Кількість угод: {len(trading_results['trades'])}")
            print(f"Комісії: ${trading_results.get('total_fees', 0):.2f}")
            print(f"Відсоток успішних угод: {trading_results.get('win_rate', 0):.1f}%")
            
            if trading_results['trades']:
                buy_trades = len([t for t in trading_results['trades'] if t[0] == 'BUY'])
                sell_trades = len([t for t in trading_results['trades'] if t[0] == 'SELL'])
                print(f"Купівлі: {buy_trades}, Продажі: {sell_trades}")
                
                print(f"\nОстанні 5 угод:")
                for trade in trading_results['trades'][-5:]:
                    trade_type, price, amount, date = trade
                    action = "Купівля" if trade_type == 'BUY' else "Продаж"
                    total_value = price * amount
                    print(f"  {action}: {amount:.4f} монет за ${price:.2f} (Вартість: ${total_value:.2f}) ({date.strftime('%Y-%m-%d')})")
                    
                # Додаткова аналітика
                first_trade_date = trading_results['trades'][0][3] if trading_results['trades'] else None
                last_trade_date = trading_results['trades'][-1][3] if trading_results['trades'] else None
                if first_trade_date and last_trade_date:
                    trading_period = (last_trade_date - first_trade_date).days
                    print(f"Період торгівлі: {trading_period} днів")
                    if trading_period > 0:
                        trades_per_day = len(trading_results['trades']) / trading_period
                        print(f"Угод на день: {trades_per_day:.2f}")
            else:
                print("❌ Торгівля не відбувалась. Можливі причини:")
                print("   - Занадто жорсткі умови торгівлі")
                print("   - Недостатньо волатильності в тестових даних")
                print("   - Прогнози не показують достатніх змін")
        
        # Додавання метрик якості моделі
        print(f"\n=== ЯКІСТЬ МОДЕЛІ ===")
        if hasattr(self, 'metrics'):
            print(f"RMSE (навчання): {self.metrics['train_rmse']:.2f}")
            print(f"RMSE (валідація): {self.metrics['val_rmse']:.2f}")
            print(f"RMSE (тест): {self.metrics['test_rmse']:.2f}")
            print(f"R² (тест): {self.metrics['test_r2']:.4f}")
            
        if hasattr(self, 'cv_scores'):
            print(f"Cross-validation RMSE: {np.mean(self.cv_scores):.2f} (±{np.std(self.cv_scores):.2f})")
            
        # Додавання трендового аналізу
        recent_prices = self.df['close'].tail(10).values
        trend = "зростання" if recent_prices[-1] > recent_prices[0] else "спадання"
        volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
        
        print(f"\n=== РИНКОВИЙ АНАЛІЗ ===")
        print(f"Поточний тренд (останні 10 днів): {trend}")
        print(f"Волатильність: {volatility:.2f}%")
        print(f"Мін. ціна за останні 30 днів: ${self.df['close'].tail(30).min():.2f}")
        print(f"Макс. ціна за останні 30 днів: ${self.df['close'].tail(30).max():.2f}")
        
        # Рекомендації
        confidence = abs(self.metrics['test_r2']) if hasattr(self, 'metrics') else 0
        if confidence > 0.7:
            recommendation = "ВИСОКА довіра до прогнозу"
        elif confidence > 0.4:
            recommendation = "СЕРЕДНЯ довіра до прогнозу"
        else:
            recommendation = "НИЗЬКА довіра до прогнозу"
            
        print(f"\n=== РЕКОМЕНДАЦІЯ ===")
        print(f"Рівень довіри: {recommendation}")
        if change_tomorrow > 5:
            print("⚠️  Очікується значне зростання - можливо варто купувати")
        elif change_tomorrow < -5:
            print("⚠️  Очікується значне падіння - можливо варто продавати")
        else:
            print("📊 Очікуються помірні зміни - утримувати позицію")

    def adaptive_trading_thresholds(self, prices, base_threshold=0.1):
        """Адаптивні пороги торгівлі базовані на волатільності ринку"""
        if len(prices) < 20:
            return base_threshold, base_threshold
            
        # Розрахунок волатільності
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100
        
        # Адаптивні пороги: більша волатільність = менші пороги
        if volatility > 5:  # Висока волатільність
            buy_threshold = base_threshold * 0.5
            sell_threshold = base_threshold * 0.5
        elif volatility > 2:  # Середня волатільність
            buy_threshold = base_threshold * 0.7
            sell_threshold = base_threshold * 0.7
        else:  # Низька волатільність
            buy_threshold = base_threshold
            sell_threshold = base_threshold
            
        return buy_threshold, sell_threshold

    def simulate_trading_all_periods(self, initial_balance=100.0):
        """Торгівля на всіх періодах даних для навчання моделі"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        # Отримуємо всі передбачення для всіх періодів
        train_start_idx = self.window_size
        val_start_idx = train_start_idx + len(self.train_pred)
        test_start_idx = val_start_idx + len(self.val_pred)
        
        # Об'єднуємо всі передбачення та ціни
        all_predictions = np.concatenate([
            self.train_pred.flatten(),
            self.val_pred.flatten(), 
            self.test_pred.flatten()
        ])
        
        all_prices = self.df['close'].iloc[train_start_idx:test_start_idx + len(self.test_pred)]
        
        balance = initial_balance
        position = 0
        trades = []
        balances = [initial_balance]
        
        print(f"🔄 Торгівля на {len(all_predictions)} днях даних...")
        
        for i in range(len(all_predictions) - 1):
            current_price = all_prices.iloc[i]
            next_predicted = all_predictions[i + 1]
            
            # Визначаємо поточний період
            if i < len(self.train_pred):
                period = "TRAIN"
            elif i < len(self.train_pred) + len(self.val_pred):
                period = "VAL"
            else:
                period = "TEST"
            
            # Отримуємо історичні дані для індикаторів
            price_history = self.df['close'].iloc[:train_start_idx + i + 1].values
            
            if len(price_history) >= 20:
                # Адаптивні пороги
                buy_threshold, sell_threshold = self.adaptive_trading_thresholds(price_history, 0.05)
                
                # Технічні індикатори
                rsi, sma_short, sma_long, volatility = self.calculate_technical_indicators(price_history)
                
                prediction_change = (next_predicted - current_price) / current_price * 100
                ma_signal = sma_short > sma_long
                
                # Адаптивна логіка торгівлі
                buy_signals = 0
                sell_signals = 0
                
                # Основні сигнали передбачення
                if prediction_change > buy_threshold:
                    buy_signals += 2
                elif prediction_change < -sell_threshold:
                    sell_signals += 2
                
                # RSI сигнали (адаптивні пороги)
                rsi_oversold = 35 if volatility > 3 else 30
                rsi_overbought = 65 if volatility > 3 else 70
                
                if rsi < rsi_oversold:
                    buy_signals += 1
                elif rsi > rsi_overbought:
                    sell_signals += 1
                
                # MA тренд
                if ma_signal:
                    buy_signals += 1
                else:
                    sell_signals += 1
                
                # Волатільність-базована торгівля
                if volatility > 4:  # Висока волатільність - більш агресивна торгівля
                    if prediction_change > 0.02:  # навіть маленькі зміни
                        buy_signals += 1
                    elif prediction_change < -0.02:
                        sell_signals += 1
                
                # Торгові рішення
                if buy_signals >= 1 and position == 0 and balance > current_price:
                    # Динамічний розмір позиції
                    confidence = min(buy_signals / 3.0, 1.0)
                    investment_ratio = 0.2 + (confidence * 0.6)  # 20-80%
                    investment_amount = balance * investment_ratio
                    coins_to_buy = investment_amount / current_price
                    
                    position += coins_to_buy
                    balance -= investment_amount
                    trades.append(('BUY', current_price, coins_to_buy, all_prices.index[i], period))
                    
                elif sell_signals >= 1 and position > 0:
                    # Продаж з урахуванням сигналів
                    confidence = min(sell_signals / 3.0, 1.0)
                    sell_ratio = 0.3 + (confidence * 0.7)  # 30-100%
                    coins_to_sell = position * sell_ratio
                    
                    sell_amount = coins_to_sell * current_price
                    balance += sell_amount
                    position -= coins_to_sell
                    trades.append(('SELL', current_price, coins_to_sell, all_prices.index[i], period))
            
            # Розрахунок поточного балансу
            current_balance = balance + (position * current_price)
            balances.append(current_balance)
        
        # Фінальний продаж
        if position > 0:
            final_price = all_prices.iloc[-1]
            balance += position * final_price
            trades.append(('SELL', final_price, position, all_prices.index[-1], 'FINAL'))
            position = 0
        
        final_balance = balance
        
        # Детальна аналітика
        if trades:
            total_fees = len(trades) * 0.02  # 0.02% комісія
            final_balance -= total_fees
            
            # Аналіз за періодами
            train_trades = [t for t in trades if 'TRAIN' in str(t[4])]
            val_trades = [t for t in trades if 'VAL' in str(t[4])]
            test_trades = [t for t in trades if 'TEST' in str(t[4])]
            
            # Розрахунок win rate
            buy_trades = [t for t in trades if t[0] == 'BUY']
            sell_trades = [t for t in trades if t[0] == 'SELL']
            
            win_trades = 0
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                for i in range(min(len(buy_trades), len(sell_trades))):
                    if sell_trades[i][1] > buy_trades[i][1]:
                        win_trades += 1
                win_rate = (win_trades / min(len(buy_trades), len(sell_trades))) * 100
            else:
                win_rate = 0
        else:
            total_fees = 0
            win_rate = 0
            train_trades = val_trades = test_trades = []
        
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit_loss': final_balance - initial_balance,
            'profit_percentage': ((final_balance - initial_balance) / initial_balance) * 100,
            'trades': trades,
            'balance_history': balances,
            'all_dates': all_prices.index,
            'total_fees': total_fees,
            'win_rate': win_rate,
            'strategy': 'adaptive_all_periods',
            'train_trades': len(train_trades),
            'val_trades': len(val_trades),
            'test_trades': len(test_trades),
            'periods_traded': len(all_predictions)
        }

    def compare_strategies(self, initial_balance=100.0):
        """Порівняння різних торгових стратегій"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        strategies = ['simple', 'enhanced', 'adaptive_all_periods']
        results = {}
        
        print("\n=== ПОРІВНЯННЯ СТРАТЕГІЙ ===")
        
        for strategy in strategies:
            if strategy == 'adaptive_all_periods':
                result = self.simulate_trading_all_periods(initial_balance)
            else:
                result = self.simulate_trading(initial_balance, strategy)
            results[strategy] = result
            
            print(f"\n📊 {strategy.upper()} стратегія:")
            print(f"   Фінальний баланс: ${result['final_balance']:.2f}")
            print(f"   Прибуток: {result['profit_percentage']:+.2f}%")
            print(f"   Кількість угод: {len(result['trades'])}")
            
            if 'train_trades' in result:
                print(f"   Угоди в навчанні: {result['train_trades']}")
                print(f"   Угоди в валідації: {result['val_trades']}")
                print(f"   Угоди в тестуванні: {result['test_trades']}")
                print(f"   Всього періодів: {result['periods_traded']}")
            
            print(f"   Відсоток успішних угод: {result.get('win_rate', 0):.1f}%")
            print(f"   Комісії: ${result.get('total_fees', 0):.2f}")
            
        # Визначення найкращої стратегії
        best_strategy = max(results.keys(), key=lambda k: results[k]['final_balance'])
        print(f"\n🏆 Найкраща стратегія: {best_strategy.upper()}")
        print(f"   Прибуток: {results[best_strategy]['profit_percentage']:+.2f}%")
        
        return results

    def real_time_prediction(self, symbol="BTCUSDT", days_ahead=7):
        """Прогнозування в реальному часі для торгових рішень"""
        if self.model is None:
            print("Модель не навчена!")
            return None
            
        # Отримання останніх даних
        current_price = self.df['close'].iloc[-1]
        future_pred = self.predict_future(days_ahead)
        
        # Розрахунок технічних індикаторів для останніх даних
        recent_prices = self.df['close'].tail(20).values
        rsi, sma_short, sma_long, volatility = self.calculate_technical_indicators(recent_prices)
        
        print(f"\n=== ТОРГОВІ СИГНАЛИ ДЛЯ {symbol} ===")
        print(f"Поточна ціна: ${current_price:.2f}")
        print(f"RSI: {rsi:.1f} ({'Перепроданість' if rsi < 30 else 'Перекупленість' if rsi > 70 else 'Нормально'})")
        print(f"MA тренд: {'Висхідний' if sma_short > sma_long else 'Спадний'}")
        print(f"Волатильність: {volatility:.2f}%")
        
        # Торгові рекомендації
        prediction_change = (future_pred['predicted_price'].iloc[0] - current_price) / current_price * 100
        
        signals = []
        if prediction_change > 1:
            signals.append("📈 Прогноз зростання")
        elif prediction_change < -1:
            signals.append("📉 Прогноз падіння")
            
        if rsi < 30:
            signals.append("💰 RSI показує перепроданість")
        elif rsi > 70:
            signals.append("⚠️ RSI показує перекупленість")
            
        if sma_short > sma_long:
            signals.append("📊 Висхідний тренд")
        else:
            signals.append("📊 Спадний тренд")
            
        print(f"\nСигнали:")
        for signal in signals:
            print(f"  {signal}")
            
        # Загальна рекомендація
        buy_score = 0
        sell_score = 0
        
        if prediction_change > 0.5:
            buy_score += 2
        elif prediction_change < -0.5:
            sell_score += 2
            
        if rsi < 35:
            buy_score += 1
        elif rsi > 65:
            sell_score += 1
            
        if sma_short > sma_long:
            buy_score += 1
        else:
            sell_score += 1
            
        print(f"\n🎯 РЕКОМЕНДАЦІЯ:")
        if buy_score > sell_score:
            print("   🟢 КУПУВАТИ")
        elif sell_score > buy_score:
            print("   🔴 ПРОДАВАТИ")
        else:
            print("   🟡 УТРИМУВАТИ")
            
        return {
            'current_price': current_price,
            'prediction_change': prediction_change,
            'rsi': rsi,
            'trend': 'up' if sma_short > sma_long else 'down',
            'volatility': volatility,
            'recommendation': 'buy' if buy_score > sell_score else 'sell' if sell_score > buy_score else 'hold'
        }

# Використання
if __name__ == "__main__":
    # Створення та навчання моделі
    predictor = CryptoPricePredictor(window_size=60)
    
    # Можна змінити символ на будь-який з Binance (BTC, ETH, BNB тощо)
    symbol = "BTCUSDT"  # або "BNBUSDT", "ETHUSDT", тощо
    
    print("🚀 Запуск системи криптовалютної торгівлі...")
    
    if predictor.train_model(symbol=symbol, start_date="2020-01-01"):
        print("✅ Модель успішно навчена!")
        
        # Основна візуалізація
        predictor.plot_results(symbol=symbol)
        
        # Порівняння стратегій
        predictor.compare_strategies(initial_balance=100.0)
        
        # Торгові сигнали в реальному часі
        predictor.real_time_prediction(symbol=symbol)
        
    else:
        print("❌ Помилка при навчанні моделі!")
