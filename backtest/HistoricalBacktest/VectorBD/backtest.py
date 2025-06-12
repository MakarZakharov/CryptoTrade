import pandas as pd
import numpy as np
import os
import glob
import importlib.util
import inspect
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class VectorBTStrategy:
    """
    Векторизованная стратегия для работы с pandas DataFrame
    Автоматически конвертирует backtrader стратегии в векторизованный формат
    """
    
    def __init__(self, strategy_class, params: dict = None):
        self.strategy_class = strategy_class
        self.params = params or {}
        self.name = strategy_class.__name__
        
        # Извлекаем параметры из backtrader стратегии
        self._extract_strategy_params()
        
        # Определяем тип стратегии для правильной генерации сигналов
        self.strategy_type = self._identify_strategy_type()
        
    def _extract_strategy_params(self):
        """Улучшенное извлечение параметров из backtrader стратегий"""
        
        # Базовые параметры по умолчанию
        base_params = {
            'position_size': 1.0,
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'sma_fast': 10, 'sma_slow': 20, 'sma_period': 20,
            'ema_fast': 12, 'ema_slow': 26,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bb_period': 20, 'bb_dev': 2, 'bb_devfactor': 2,
            'momentum_period': 14, 'momentum_threshold': 0.02,
            'stop_loss': 0.05, 'take_profit': 0.15,
            'ma_fast': 10, 'ma_slow': 30, 'ma_fast_period': 10, 'ma_slow_period': 30,
            'oversold_level': 30, 'overbought_level': 70,
            'trade_probability': 0.1, 'hold_days': 5,
            'quick_profit': 0.02, 'rsi_upper': 70, 'rsi_lower': 30,
            'rsi_exit_overbought': 75, 'rsi_exit_oversold': 25
        }
        
        # Специфичные параметры для разных стратегий
        strategy_specific = {
            'LOLStrategy': {
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
                'sma_fast': 10, 'sma_slow': 20, 'stop_loss': 0.05, 'take_profit': 0.15
            },
            'LOLScalpingStrategy': {
                'rsi_period': 7, 'rsi_oversold': 35, 'rsi_overbought': 65,
                'ema_fast': 5, 'ema_slow': 13, 'quick_profit': 0.02
            },
            'LOLRandomStrategy': {
                'trade_probability': 0.1, 'hold_days': 5
            },
            'RSI_MA_Strategy': {
                'rsi_period': 14, 'ma_fast_period': 10, 'ma_slow_period': 30,
                'oversold_level': 45, 'overbought_level': 55
            },
            'RSI_SMA_Strategy': {
                'rsi_period': 18, 'rsi_overbought': 35, 'rsi_oversold': 65,
                'sma_fast': 10, 'sma_slow': 20
            },
            'MACD_SMA_Strategy': {
                'macd_fast': 14, 'macd_slow': 20, 'sma_fast': 20, 'sma_slow': 75
            },
            'BollingerBandsStrategy': {
                'bb_period': 20, 'bb_dev': 2, 'rsi_period': 14
            },
            'MACDStrategy': {
                'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9
            },
            'ProfitableBTCStrategy': {
                'ema_fast': 12, 'ema_slow': 26, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
            },
            'SafeProfitableBTCStrategy': {
                'ema_fast': 12, 'ema_slow': 26, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70
            }
        }
        
        # Применяем базовые параметры
        self.params.update(base_params)
        
        # Применяем специфичные параметры для стратегии
        if self.name in strategy_specific:
            self.params.update(strategy_specific[self.name])
            
    def _identify_strategy_type(self):
        """Определяет тип стратегии по названию"""
        name = self.name.lower()
        
        if 'rsi' in name and 'sma' in name:
            return 'rsi_sma'
        elif 'rsi' in name and 'ma' in name:
            return 'rsi_ma'
        elif 'macd' in name and 'sma' in name:
            return 'macd_sma'
        elif 'bollinger' in name:
            return 'bollinger'
        elif 'macd' in name:
            return 'macd'
        elif 'rsi' in name:
            return 'rsi'
        elif 'movingaverage' in name or name.endswith('crossover') or name.endswith('cross'):
            return 'ma_cross'
        elif 'momentum' in name:
            return 'momentum'
        elif 'hybrid' in name:
            return 'hybrid'
        elif 'lolrandom' in name:
            return 'lol_random'
        elif 'lolscalping' in name:
            return 'lol_scalping'
        elif 'lol' in name:
            return 'lol'
        elif 'scalping' in name:
            return 'scalping'
        elif 'profitable' in name or 'btc' in name:
            return 'btc_strategy'
        else:
            return 'general'
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет индикаторы для всего датасета векторизованно"""
        df = data.copy()
        
        # RSI
        if any(param in self.params for param in ['rsi_period', 'rsi_oversold', 'rsi_overbought']):
            period = self.params.get('rsi_period', 14)
            df['rsi'] = self._calculate_rsi(df['close'], period)
        
        # SMA
        if any(param in self.params for param in ['sma_fast', 'sma_slow', 'sma_period']):
            if 'sma_fast' in self.params:
                df['sma_fast'] = df['close'].rolling(self.params['sma_fast']).mean()
            if 'sma_slow' in self.params:
                df['sma_slow'] = df['close'].rolling(self.params['sma_slow']).mean()
            if 'sma_period' in self.params:
                df['sma'] = df['close'].rolling(self.params['sma_period']).mean()
        
        # EMA
        if any(param in self.params for param in ['ema_fast', 'ema_slow']):
            if 'ema_fast' in self.params:
                df['ema_fast'] = df['close'].ewm(span=self.params['ema_fast']).mean()
            if 'ema_slow' in self.params:
                df['ema_slow'] = df['close'].ewm(span=self.params['ema_slow']).mean()
        
        # MACD
        if any(param in self.params for param in ['macd_fast', 'macd_slow', 'macd_signal']):
            fast = self.params.get('macd_fast', 12)
            slow = self.params.get('macd_slow', 26)
            signal = self.params.get('macd_signal', 9)
            
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if any(param in self.params for param in ['bb_period', 'bb_dev']):
            period = self.params.get('bb_period', 20)
            dev = self.params.get('bb_dev', 2)
            
            df['bb_middle'] = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * dev)
            df['bb_lower'] = df['bb_middle'] - (bb_std * dev)
        
        # Momentum
        if 'momentum_period' in self.params:
            period = self.params['momentum_period']
            df['momentum'] = df['close'].diff(period)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Векторизованное вычисление RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Генерирует торговые сигналы на основе стратегии"""
        df = self.calculate_indicators(data)
        
        # Инициализируем сигналы
        df['buy_signal'] = False
        df['sell_signal'] = False
        
        # Используем тип стратегии для точной генерации сигналов
        if self.strategy_type == 'rsi_sma':
            self._rsi_sma_signals(df)
        elif self.strategy_type == 'rsi_ma':
            self._rsi_ma_signals(df)
        elif self.strategy_type == 'macd_sma':
            self._macd_sma_signals(df)
        elif self.strategy_type == 'bollinger':
            self._bollinger_signals(df)
        elif self.strategy_type == 'macd':
            self._macd_signals(df)
        elif self.strategy_type == 'rsi':
            self._rsi_signals(df)
        elif self.strategy_type == 'ma_cross':
            self._ma_cross_signals(df)
        elif self.strategy_type == 'momentum':
            self._momentum_signals(df)
        elif self.strategy_type == 'hybrid':
            self._hybrid_signals(df)
        elif self.strategy_type == 'lol_random':
            self._lol_random_signals(df)
        elif self.strategy_type == 'lol_scalping':
            self._lol_scalping_signals(df)
        elif self.strategy_type == 'lol':
            self._lol_signals(df)
        elif self.strategy_type == 'scalping':
            self._scalping_signals(df)
        elif self.strategy_type == 'btc_strategy':
            self._btc_strategy_signals(df)
        else:
            self._general_signals(df)
        
        return df
    
    def _rsi_signals(self, df: pd.DataFrame):
        """RSI сигналы"""
        oversold = self.params.get('rsi_oversold', 30)
        overbought = self.params.get('rsi_overbought', 70)
        
        if 'rsi' in df.columns:
            df['buy_signal'] = df['rsi'] < oversold
            df['sell_signal'] = df['rsi'] > overbought
        else:
            df['buy_signal'] = False
            df['sell_signal'] = False
    
    def _macd_signals(self, df: pd.DataFrame):
        """MACD сигналы"""
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            # Пересечение MACD и сигнальной линии
            df['buy_signal'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            df['sell_signal'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    def _bollinger_signals(self, df: pd.DataFrame):
        """Bollinger Bands сигналы"""
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            df['buy_signal'] = df['close'] <= df['bb_lower']
            df['sell_signal'] = df['close'] >= df['bb_upper']
    
    def _ma_cross_signals(self, df: pd.DataFrame):
        """Moving Average Cross сигналы"""
        if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
            # Пересечение скользящих средних
            df['buy_signal'] = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
            df['sell_signal'] = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))
        elif 'ema_fast' in df.columns and 'ema_slow' in df.columns:
            # Пересечение EMA
            df['buy_signal'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
            df['sell_signal'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
    
    def _rsi_sma_signals(self, df: pd.DataFrame):
        """RSI + SMA комбинированные сигналы"""
        if 'rsi' in df.columns:
            rsi_overbought = self.params.get('rsi_overbought', 35)
            rsi_oversold = self.params.get('rsi_oversold', 65)
            
            # Агрессивная RSI_SMA_Strategy логика
            long_conditions = [
                df['rsi'] < rsi_oversold,
                (df['rsi'] < 35) & (df['rsi'] > df['rsi'].shift(1)),  # RSI растет от низких значений
                df['close'] > df['close'].shift(1),  # Цена растет
                (df['rsi'] > 30) & (df['rsi'] < 50),  # RSI в зоне восстановления
            ]
            
            short_conditions = [
                df['rsi'] > rsi_overbought,
                (df['rsi'] > 65) & (df['rsi'] < df['rsi'].shift(1)),  # RSI падает от высоких значений
                df['close'] < df['close'].shift(1),  # Цена падает
                (df['rsi'] < 70) & (df['rsi'] > 50),  # RSI в зоне ослабления
            ]
            
            # SMA сигналы если доступны
            if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
                sma_cross_up = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
                sma_cross_down = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))
                long_conditions.extend([sma_cross_up, df['close'] > df['sma_fast']])
                short_conditions.extend([sma_cross_down, df['close'] < df['sma_fast']])
            
            # Выполняем сделку если выполнено хотя бы 2 условия
            df['buy_signal'] = pd.concat(long_conditions, axis=1).sum(axis=1) >= 2
            df['sell_signal'] = pd.concat(short_conditions, axis=1).sum(axis=1) >= 2
    
    def _rsi_ma_signals(self, df: pd.DataFrame):
        """RSI + MA стратегия сигналы"""
        if 'rsi' in df.columns:
            oversold = self.params.get('oversold_level', 45)
            overbought = self.params.get('overbought_level', 55)
            
            # Упрощенные условия для покупки
            df['buy_signal'] = df['rsi'] < oversold
            
            # Условия для продажи
            sell_conditions = [
                df['rsi'] > overbought,
            ]
            
            # Добавляем MA условия если доступны
            if 'ma_fast_period' in self.params and 'sma_fast' in df.columns and 'sma_slow' in df.columns:
                sell_conditions.append(df['sma_fast'] < df['sma_slow'])  # Тренд изменился
            
            df['sell_signal'] = pd.concat(sell_conditions, axis=1).any(axis=1)
    
    def _macd_sma_signals(self, df: pd.DataFrame):
        """MACD + SMA комбинированные сигналы"""
        if all(col in df.columns for col in ['macd', 'macd_signal', 'sma_fast', 'sma_slow']):
            # Покупка: MACD пересекает сигнальную линию вверх + быстрая SMA выше медленной
            macd_cross_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            sma_bullish = df['sma_fast'] > df['sma_slow']
            df['buy_signal'] = macd_cross_up & sma_bullish
            
            # Продажа: MACD пересекает сигнальную линию вниз + быстрая SMA ниже медленной
            macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            sma_bearish = df['sma_fast'] < df['sma_slow']
            df['sell_signal'] = macd_cross_down & sma_bearish
    
    def _lol_random_signals(self, df: pd.DataFrame):
        """LOL Random стратегия сигналы"""
        np.random.seed(42)  # Для воспроизводимости
        prob = self.params.get('trade_probability', 0.1)
        df['buy_signal'] = np.random.random(len(df)) < prob
        df['sell_signal'] = False  # Продаем через N дней в бэктестере
    
    def _lol_scalping_signals(self, df: pd.DataFrame):
        """LOL Scalping стратегия сигналы"""
        if 'rsi' in df.columns and 'ema_fast' in df.columns and 'ema_slow' in df.columns:
            rsi_oversold = self.params.get('rsi_oversold', 35)
            rsi_overbought = self.params.get('rsi_overbought', 65)
            
            # Быстрые сигналы на вход
            ema_cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
            df['buy_signal'] = ema_cross_up & (df['rsi'] < rsi_oversold)
            
            # Быстрый выход
            ema_cross_down = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
            df['sell_signal'] = ema_cross_down | (df['rsi'] > rsi_overbought)
    
    def _scalping_signals(self, df: pd.DataFrame):
        """Scalping стратегия сигналы"""
        if 'rsi' in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            rsi_upper = self.params.get('rsi_upper', 65)
            rsi_lower = self.params.get('rsi_lower', 35)
            
            # Сигналы на покупку (очень агрессивные)
            buy_cond = (df['rsi'] < rsi_lower) & (df['close'] <= df['bb_lower'])
            if 'ema_fast' in df.columns:
                buy_cond &= (df['close'] > df['ema_fast'])
            df['buy_signal'] = buy_cond
            
            # Сигналы на продажу
            sell_cond = (df['rsi'] > rsi_upper) & (df['close'] >= df['bb_upper'])
            if 'ema_fast' in df.columns:
                sell_cond |= (df['close'] < df['ema_fast'])
            df['sell_signal'] = sell_cond
    
    def _btc_strategy_signals(self, df: pd.DataFrame):
        """BTC стратегия сигналы (Profitable/Safe)"""
        if 'rsi' in df.columns and 'ema_fast' in df.columns and 'ema_slow' in df.columns:
            rsi_oversold = self.params.get('rsi_oversold', 30)
            rsi_overbought = self.params.get('rsi_overbought', 70)
            
            # Сигналы на покупку
            ema_bullish = df['ema_fast'] > df['ema_slow']
            rsi_oversold_signal = df['rsi'] < rsi_oversold
            ema_cross_up = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
            
            df['buy_signal'] = ema_bullish | rsi_oversold_signal | ema_cross_up
            
            # Сигналы на продажу
            rsi_overbought_signal = df['rsi'] > rsi_overbought
            ema_bearish = ~ema_bullish
            ema_cross_down = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
            
            df['sell_signal'] = (rsi_overbought_signal & ema_bearish) | ema_cross_down
    
    def _momentum_signals(self, df: pd.DataFrame):
        """Momentum сигналы"""
        if 'momentum' in df.columns:
            threshold = self.params.get('momentum_threshold', 0.02)
            momentum_pct = df['momentum'] / df['close']
            df['buy_signal'] = momentum_pct > threshold
            df['sell_signal'] = momentum_pct < -threshold
    
    def _hybrid_signals(self, df: pd.DataFrame):
        """Hybrid стратегия сигналы"""
        oversold = self.params.get('rsi_oversold', 35)
        overbought = self.params.get('rsi_overbought', 65)
        
        if 'rsi' in df.columns and 'sma' in df.columns:
            df['buy_signal'] = (df['close'] > df['sma']) & (df['rsi'] < oversold)
            df['sell_signal'] = (df['close'] < df['sma']) | (df['rsi'] > overbought)
    
    def _lol_signals(self, df: pd.DataFrame):
        """LOL стратегии сигналы"""
        if 'rsi' in df.columns:
            oversold = self.params.get('rsi_oversold', 30)
            overbought = self.params.get('rsi_overbought', 70)
            
            buy_condition = df['rsi'] < oversold
            if 'sma_fast' in df.columns:
                buy_condition &= (df['close'] > df['sma_fast'])
            
            sell_condition = df['rsi'] > overbought
            if 'sma_slow' in df.columns:
                sell_condition |= (df['close'] < df['sma_slow'])
            
            df['buy_signal'] = buy_condition
            df['sell_signal'] = sell_condition
    
    def _general_signals(self, df: pd.DataFrame):
        """Общие сигналы для неопознанных стратегий"""
        # Комбинированная логика
        buy_conditions = []
        sell_conditions = []
        
        if 'rsi' in df.columns:
            oversold = self.params.get('rsi_oversold', 30)
            overbought = self.params.get('rsi_overbought', 70)
            buy_conditions.append(df['rsi'] < oversold)
            sell_conditions.append(df['rsi'] > overbought)
        
        if 'sma_fast' in df.columns and 'sma_slow' in df.columns:
            buy_conditions.append((df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1)))
            sell_conditions.append((df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1)))
        
        if buy_conditions:
            df['buy_signal'] = pd.concat(buy_conditions, axis=1).any(axis=1)
        else:
            df['buy_signal'] = False
            
        if sell_conditions:
            df['sell_signal'] = pd.concat(sell_conditions, axis=1).any(axis=1)
        else:
            df['sell_signal'] = False


class VectorBTBacktester:
    """
    Векторизованный бэктестер использующий pandas для высокой производительности
    """
    
    def __init__(self, initial_cash: float = 100000, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        self.results = []
        
    def load_strategies(self, strategy_dir: str) -> List[Any]:
        """Загружает стратегии из директории"""
        strategies = []
        
        for file_path in glob.glob(os.path.join(strategy_dir, "*.py")):
            if file_path.endswith("__init__.py"):
                continue
                
            try:
                spec = importlib.util.spec_from_file_location("strategy_module", file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        name.endswith('Strategy') and 
                        hasattr(obj, 'params')):
                        strategies.append(obj)
                        
            except Exception as e:
                print(f"⚠️ Ошибка загрузки {file_path}: {e}")
                
        return strategies
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Загружает исторические данные"""
        try:
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"❌ Ошибка загрузки данных {data_path}: {e}")
            return None
    
    def backtest_strategy(self, strategy: VectorBTStrategy, data: pd.DataFrame) -> Dict:
        """Выполняет бэктест стратегии"""
        # Генерируем сигналы
        df = strategy.generate_signals(data)
        
        # Инициализируем портфель
        df['position'] = 0
        df['cash'] = self.initial_cash
        df['holdings'] = 0
        df['total'] = self.initial_cash
        
        position = 0
        cash = self.initial_cash
        trades = []
        entry_price = 0
        
        position_size = strategy.params.get('position_size', 1.0)
        
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            
            # Обновляем стоимость портфеля
            holdings = position * current_price
            total = cash + holdings
            
            # Проверяем сигналы
            if position == 0 and df['buy_signal'].iloc[i]:
                # Покупаем
                trade_size = (cash * position_size) / current_price
                cost = trade_size * current_price
                commission_cost = cost * self.commission
                
                if cash >= cost + commission_cost:
                    cash -= (cost + commission_cost)
                    position = trade_size
                    entry_price = current_price
                    
                    trades.append({
                        'type': 'BUY',
                        'timestamp': df.index[i],
                        'price': current_price,
                        'size': trade_size,
                        'cost': cost + commission_cost
                    })
            
            elif position > 0 and df['sell_signal'].iloc[i]:
                # Продаем
                revenue = position * current_price
                commission_cost = revenue * self.commission
                
                cash += (revenue - commission_cost)
                pnl = revenue - (position * entry_price) - (commission_cost * 2)
                
                trades.append({
                    'type': 'SELL',
                    'timestamp': df.index[i],
                    'price': current_price,
                    'size': position,
                    'revenue': revenue - commission_cost,
                    'pnl': pnl
                })
                
                position = 0
                entry_price = 0
            
            # Проверяем стоп-лосс и тейк-профит
            if position > 0 and entry_price > 0:
                stop_loss = strategy.params.get('stop_loss', 0.10)
                take_profit = strategy.params.get('take_profit', 0.20)
                
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                    # Принудительно закрываем позицию
                    revenue = position * current_price
                    commission_cost = revenue * self.commission
                    
                    cash += (revenue - commission_cost)
                    pnl = revenue - (position * entry_price) - (commission_cost * 2)
                    
                    trades.append({
                        'type': 'STOP/PROFIT',
                        'timestamp': df.index[i],
                        'price': current_price,
                        'size': position,
                        'revenue': revenue - commission_cost,
                        'pnl': pnl,
                        'reason': 'STOP_LOSS' if pnl_pct <= -stop_loss else 'TAKE_PROFIT'
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Сохраняем состояние портфеля
            df.loc[df.index[i], 'position'] = position
            df.loc[df.index[i], 'cash'] = cash
            df.loc[df.index[i], 'holdings'] = holdings
            df.loc[df.index[i], 'total'] = total
        
        # Рассчитываем финальные метрики
        final_value = df['total'].iloc[-1]
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / max(len(trades), 1) * 100 if trades else 0
        
        # Максимальная просадка
        df['peak'] = df['total'].expanding().max()
        df['drawdown'] = (df['total'] - df['peak']) / df['peak'] * 100
        max_drawdown = df['drawdown'].min()
        
        return {
            'strategy_name': strategy.name,
            'parameters': strategy.params,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': df[['total', 'cash', 'holdings']].copy()
        }
    
    def run_backtest(self, strategies: List[Any], data_files: List[str]) -> None:
        """Запускает бэктест для всех стратегий и данных"""
        self.results = []
        
        print("🚀 ВЕКТОРИЗОВАННЫЙ БЭКТЕСТЕР")
        print("=" * 80)
        
        for data_file in data_files:
            print(f"\n📈 Тестирование на: {os.path.basename(data_file)}")
            print("-" * 50)
            
            data = self.load_data(data_file)
            if data is None:
                continue
            
            # Обрабатываем все стратегии
            for i, strategy_class in enumerate(strategies, 1):
                try:
                    # Создаем векторизованную стратегию
                    vector_strategy = VectorBTStrategy(strategy_class)
                    
                    # Запускаем бэктест
                    result = self.backtest_strategy(vector_strategy, data)
                    result['data_file'] = os.path.basename(data_file)
                    
                    self.results.append(result)
                    
                    # Выводим результаты
                    print(f"✅ {i:2d}. {result['strategy_name']} [{vector_strategy.strategy_type}]")
                    print(f"      💰 Финальная стоимость: ${result['final_value']:,.2f}")
                    print(f"      📊 Доходность: {result['total_return']:+.2f}%")
                    print(f"      🎯 Сделок: {result['total_trades']}")
                    print(f"      🏆 Процент побед: {result['win_rate']:.1f}%")
                    print(f"      📉 Макс. просадка: {result['max_drawdown']:.2f}%")
                    print()
                    
                except Exception as e:
                    print(f"❌ {i:2d}. Ошибка в {strategy_class.__name__}: {e}")
                    import traceback
                    traceback.print_exc()
    
    def get_best_strategies(self, top_n: int = 5) -> List[Dict]:
        """Возвращает топ лучших стратегий по доходности"""
        sorted_results = sorted(self.results, key=lambda x: x['total_return'], reverse=True)
        return sorted_results[:top_n]
    
    def print_summary(self):
        """Выводит сводку результатов"""
        if not self.results:
            print("❌ Нет результатов для отображения")
            return
        
        print("\n" + "=" * 80)
        print("📊 СВОДКА РЕЗУЛЬТАТОВ")
        print("=" * 80)
        
        best_strategies = self.get_best_strategies(10)
        
        for i, result in enumerate(best_strategies, 1):
            print(f"{i:2d}. 🏆 {result['strategy_name']}")
            print(f"    📂 Данные: {result['data_file']}")
            print(f"    💰 Доходность: {result['total_return']:+.2f}%")
            print(f"    🎯 Сделок: {result['total_trades']} | Побед: {result['win_rate']:.1f}%")
            print(f"    📉 Макс. просадка: {result['max_drawdown']:.2f}%")
            print()


def main():
    """Основная функция для запуска VectorBT бэктестера"""
    
    # Инициализация бэктестера
    backtester = VectorBTBacktester(initial_cash=100000, commission=0.001)
    
    # Загружаем стратегии
    strategy_dir = os.path.join(os.path.dirname(__file__), "../../../strategies/TestStrategies")
    strategies = backtester.load_strategies(strategy_dir)
    
    print(f"✅ Загружено стратегий: {len(strategies)}")
    for strategy in strategies:
        print(f"   - {strategy.__name__}")
    
    # Загружаем данные
    data_dir = os.path.join(os.path.dirname(__file__), "../../../data/binance")
    data_files = []
    
    # Собираем все CSV файлы
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                data_files.append(os.path.join(root, file))
    
    # Ограничиваем количество файлов для быстрого тестирования
    test_files = [f for f in data_files if 'BTCUSDT' in f and '1d' in f][:2]
    
    print(f"✅ Найдено файлов данных: {len(test_files)}")
    for file in test_files:
        print(f"   - {os.path.basename(file)}")
    
    # Запускаем бэктест
    backtester.run_backtest(strategies, test_files)
    
    # Показываем сводку
    backtester.print_summary()


if __name__ == "__main__":
    main()