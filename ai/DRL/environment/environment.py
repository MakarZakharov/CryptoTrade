"""
Максимально реалистичная торговая среда для обучения DRL агентов
Основана на лучших практиках из FinRL, Pro Trader RL и современных исследований

Ключевые особенности:
- Реалистичное моделирование микроструктуры рынка
- GPU-оптимизированная векторизация (как в FinRL)
- Продвинутое моделирование исполнения ордеров
- Множественные схемы вознаграждений
- Моделирование рыночного воздействия и ликвидности
- Поддержка различных типов ордеров
- Интеграция технических индикаторов и альтернативных данных
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import sys
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import deque
import random
import torch
import numba
from numba import njit
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Импорт системы вознаграждений из отдельного модуля
try:
    from .reword import (
        AdvancedRewardScheme, 
        CurriculumRewardScheme, 
        SimpleRewardScheme,
        create_reward_scheme,
        Trade,
        Position,
        OrderSide,
        OrderType
    )
except ImportError:
    # Fallback для случаев когда модуль запускается напрямую
    from reword import (
        AdvancedRewardScheme, 
        CurriculumRewardScheme, 
        SimpleRewardScheme,
        create_reward_scheme,
        Trade,
        Position,
        OrderSide,
        OrderType
    )

try:
    from CryptoTrade.ai.ML1.market_analysis.data.features.technical_indicators import TechnicalIndicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    TECHNICAL_INDICATORS_AVAILABLE = False
    logger.info("Technical indicators module not available, using built-in indicators")

try:
    from CryptoTrade.ai.ML1.market_analysis.data.fetchers.csv_fetcher import CSVFetcher
    CSV_FETCHER_AVAILABLE = True
except ImportError:
    CSV_FETCHER_AVAILABLE = False
    logger.info("CSV Fetcher module not available, using synthetic data")

# Классы Trade, Position, OrderSide, OrderType теперь импортируются из reword.py

@dataclass
class OrderBookLevel:
    """Уровень книги ордеров"""
    price: float
    quantity: float
    orders_count: int = 1

@njit
def calculate_slippage_vectorized(order_sizes: np.ndarray, 
                                volumes: np.ndarray, 
                                volatilities: np.ndarray,
                                base_slippage: float = 0.0005) -> np.ndarray:
    """Векторизованный расчет slippage с GPU оптимизацией"""
    volume_ratios = order_sizes / np.maximum(volumes, 1.0)
    impact = 0.0001 * np.sqrt(volume_ratios) * volatilities
    slippage = base_slippage * (1 + volatilities * 5 + volume_ratios * 0.1)
    return np.clip(slippage + impact, 0, 0.02)

@njit
def calculate_partial_fill_probability(order_size: float, 
                                     available_liquidity: float,
                                     volatility: float) -> float:
    """Расчет вероятности частичного исполнения ордера"""
    if order_size <= available_liquidity * 0.1:
        return 1.0  # Мгновенное исполнение для малых ордеров
    elif order_size <= available_liquidity * 0.5:
        return 0.8 + 0.2 * (1 - volatility)  # Высокая вероятность
    elif order_size <= available_liquidity:
        return 0.4 + 0.4 * (1 - volatility)  # Средняя вероятность
    else:
        return 0.1 + 0.2 * (1 - volatility)  # Низкая вероятность

class AdvancedLiquidityModel:
    """Продвинутая модель ликвидности с реалистичной микроструктурой рынка"""
    
    def __init__(self, 
                 base_spread: float = 0.001,
                 impact_factor: float = 0.0001,
                 volume_factor: float = 0.5,
                 enable_partial_fills: bool = True,
                 max_order_book_levels: int = 20):
        self.base_spread = base_spread
        self.impact_factor = impact_factor
        self.volume_factor = volume_factor
        self.enable_partial_fills = enable_partial_fills
        self.max_order_book_levels = max_order_book_levels
        
        # Исторические данные для моделирования
        self.spread_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        
        logger.info(f"Инициализирована продвинутая модель ликвидности с {max_order_book_levels} уровнями")
        
    def calculate_realistic_market_impact(self, 
                                        order_size: float, 
                                        current_volume: float,
                                        volatility: float,
                                        bid_ask_spread: float) -> Tuple[float, float]:
        """Расчет реалистичного рыночного воздействия и времени исполнения"""
        # Используем векторизованную функцию для производительности
        order_sizes = np.array([order_size])
        volumes = np.array([current_volume])
        volatilities = np.array([volatility])
        
        impact = calculate_slippage_vectorized(order_sizes, volumes, volatilities)[0]
        
        # Добавляем влияние спреда
        spread_impact = bid_ask_spread * 0.5  # Половина спреда как дополнительные издержки
        total_impact = impact + spread_impact
        
        # Время исполнения зависит от размера ордера
        volume_ratio = order_size / max(current_volume, 1.0)
        execution_time = 1 + int(volume_ratio * 10)  # От 1 до 11 тиков
        
        return min(total_impact, 0.02), execution_time  # Максимум 2% impact
    
    def simulate_partial_fill(self, 
                            order_size: float,
                            available_liquidity: float,
                            volatility: float) -> Tuple[float, bool]:
        """Симуляция частичного исполнения ордера"""
        if not self.enable_partial_fills:
            return order_size, True
            
        fill_probability = calculate_partial_fill_probability(
            order_size, available_liquidity, volatility
        )
        
        if np.random.random() < fill_probability:
            # Полное исполнение
            return order_size, True
        else:
            # Частичное исполнение
            fill_ratio = np.random.uniform(0.3, 0.8)  # Исполняется 30-80%
            filled_size = order_size * fill_ratio
            return filled_size, False
    
    def generate_realistic_order_book(self, 
                                    mid_price: float, 
                                    volume: float,
                                    volatility: float,
                                    time_of_day: float = 0.5) -> Tuple[List[OrderBookLevel], List[OrderBookLevel]]:
        """Генерация реалистичной книги ордеров с учетом времени суток"""
        # Динамический спред зависит от волатильности и времени
        base_spread = self.base_spread
        time_factor = 1.0 + 0.3 * abs(time_of_day - 0.5)  # Больше спред в нетрадиционные часы
        spread = base_spread * (1 + volatility * 10) * time_factor
        
        # Сохраняем историю для адаптивности
        self.spread_history.append(spread)
        self.volume_history.append(volume)
        
        bids = []
        asks = []
        
        levels = min(self.max_order_book_levels, 20)
        
        for i in range(levels):
            # Более реалистичное распределение ликвидности
            # Используем экспоненциальное затухание с шумом
            decay_factor = np.exp(-i * 0.25)
            noise_factor = 0.7 + 0.6 * np.random.random()
            level_volume = volume * decay_factor * noise_factor
            
            # Неравномерные шаги цен (реалистично для реальных книг ордеров)
            price_step = spread * (0.4 + i * 0.15 + np.random.normal(0, 0.05))
            
            bid_price = mid_price - price_step
            ask_price = mid_price + price_step
            
            # Количество ордеров на уровне
            orders_count = max(1, int(level_volume / (100 + np.random.exponential(50))))
            
            bids.append(OrderBookLevel(bid_price, level_volume, orders_count))
            asks.append(OrderBookLevel(ask_price, level_volume, orders_count))
            
        return bids, asks

# Класс AdvancedRewardScheme перенесен в модуль reword.py

class RealisticTradingEnvironment(gym.Env):
    """
    Максимально реалистичная торговая среда для DRL
    
    Включает:
    - Микроструктуру рынка и моделирование ликвидности
    - Реалистичное исполнение ордеров
    - Продвинутые схемы вознаграждений
    - GPU-оптимизацию (готовность к векторизации)
    - Моделирование рыночного воздействия
    - Различные типы ордеров
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Поддержка мультиактивной торговли
        self.symbols = config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.current_symbol_index = 0
        self.symbol_rotation_interval = config.get('symbol_rotation_interval', 1)  # Каждый шаг
        self.steps_since_rotation = 0
        
        self.timeframe = config.get('timeframe', '15m')  # 15-минутные данные
        self.exchange = config.get('exchange', 'binance')
        
        logger.info(f"Инициализирована мультиактивная среда с символами: {self.symbols}")
        logger.info(f"Интервал ротации: каждые {self.symbol_rotation_interval} шагов")
        
        # Торговые параметры
        self.initial_balance = config.get('initial_balance', 100000.0)
        self.commission_rate = config.get('commission_rate', 0.001)
        self.min_trade_size = config.get('min_trade_size', 10.0)
        
        # Параметры реализма
        self.enable_slippage = config.get('enable_slippage', True)
        self.enable_market_impact = config.get('enable_market_impact', True)
        self.enable_liquidity_modeling = config.get('enable_liquidity_modeling', True)
        self.enable_order_book = config.get('enable_order_book', True)
        
        # Продвинутые модели с улучшенным реализмом
        self.liquidity_model = AdvancedLiquidityModel(
            enable_partial_fills=config.get('enable_partial_fills', True),
            max_order_book_levels=config.get('max_order_book_levels', 20)
        )
        
        # Инициализация системы вознаграждений из отдельного модуля
        reward_type = config.get('reward_scheme_type', 'advanced')
        if reward_type == 'curriculum':
            # Для curriculum learning добавляем информацию об этапе обучения
            config['learning_stage'] = config.get('learning_stage', 'mixed')
        
        self.reward_scheme = create_reward_scheme(reward_type, config)
        
        # Состояние среды
        self.current_step = 0
        self.max_steps = config.get('max_steps', 10000)
        self.lookback_window = config.get('lookback_window', 50)
        
        # Состояние мультиактивной торговли
        self.positions = {}  # Позиции по каждому символу
        for symbol in self.symbols:
            self.positions[symbol] = Position(symbol, 0.0, 0.0, 0.0, 0.0, 0)
        
        # Устанавливаем текущий символ ПЕРЕД использованием
        self.current_symbol = self.symbols[self.current_symbol_index]
        
        # Торговое состояние
        self.balance = self.initial_balance
        self.position = self.positions[self.current_symbol]  # Текущая активная позиция
        self.trades_history = []
        self.portfolio_history = []
        
        # Загрузка данных для всех символов
        self.multi_symbol_data = self._load_multi_symbol_data()
        if not self.multi_symbol_data or all(df.empty for df in self.multi_symbol_data.values()):
            raise ValueError(f"No data loaded for symbols: {self.symbols}")
        
        # Текущие данные (для активного символа)
        self.data = self.multi_symbol_data[self.symbols[self.current_symbol_index]]
        self.current_symbol = self.symbols[self.current_symbol_index]
        
        logger.info(f"Загружены данные для {len(self.multi_symbol_data)} символов")
        for symbol, df in self.multi_symbol_data.items():
            logger.info(f"  {symbol}: {len(df)} записей")
            
        # Пространства
        self._setup_spaces()
        
        # Метрики
        self.reset_metrics()
        
    def _load_multi_symbol_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка данных для всех символов"""
        multi_data = {}
        
        for symbol in self.symbols:
            logger.info(f"Загрузка данных для {symbol}...")
            
            try:
                # Загружаем данные для каждого символа
                symbol_data = self._load_single_symbol_data(symbol)
                if not symbol_data.empty:
                    multi_data[symbol] = symbol_data
                    logger.info(f"✅ {symbol}: загружено {len(symbol_data)} записей")
                else:
                    # Если данные пусты, создаем синтетические
                    logger.warning(f"⚠️ {symbol}: данные пусты, создаем синтетические")
                    symbol_data = self._generate_synthetic_data_for_symbol(symbol)
                    multi_data[symbol] = symbol_data
                    logger.info(f"🔄 {symbol}: использованы синтетические данные ({len(symbol_data)} записей)")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки данных для {symbol}: {e}")
                # Генерируем синтетические данные как fallback
                try:
                    symbol_data = self._generate_synthetic_data_for_symbol(symbol)
                    if not symbol_data.empty:
                        multi_data[symbol] = symbol_data
                        logger.info(f"🔄 {symbol}: использованы синтетические данные ({len(symbol_data)} записей)")
                    else:
                        logger.error(f"❌ {symbol}: не удалось создать синтетические данные")
                        # Создаем минимальные данные
                        symbol_data = self._create_minimal_data(symbol)
                        multi_data[symbol] = symbol_data
                        logger.info(f"🔄 {symbol}: использованы минимальные данные ({len(symbol_data)} записей)")
                except Exception as e2:
                    logger.error(f"❌ {symbol}: ошибка создания синтетических данных: {e2}")
                    # Создаем минимальные данные
                    symbol_data = self._create_minimal_data(symbol)
                    multi_data[symbol] = symbol_data
                    logger.info(f"🔄 {symbol}: использованы минимальные данные ({len(symbol_data)} записей)")
        
        return multi_data
    
    def _load_single_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Загрузка и подготовка данных с техническими индикаторами для одного символа"""
        try:
            # Пытаемся загрузить реальные данные только если CSV_FETCHER доступен
            if CSV_FETCHER_AVAILABLE:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                data_path = os.path.join(project_root, 'data', self.exchange)
                
                fetcher = CSVFetcher(
                    symbol=symbol,
                    interval=self.timeframe,
                    base_path=data_path
                )
                
                data = fetcher.fetch_data(
                    start_date='2020-01-01',
                    end_date='2024-12-31'
                )
                
                if not data.empty:
                    # Добавляем технические индикаторы
                    if TECHNICAL_INDICATORS_AVAILABLE:
                        data = TechnicalIndicators.add_all_indicators(data)
                    else:
                        data = self._add_simple_technical_indicators(data)
                    
                    # Добавляем дополнительные реалистичные features
                    data = self._add_market_microstructure_features(data)
                    
                    # Добавляем информацию о символе
                    data['symbol'] = symbol
                    data['symbol_encoded'] = self.symbols.index(symbol)
                    
                    return data.fillna(0)
            
            # Если CSVFetcher недоступен или данные пусты, возвращаем пустой DataFrame
            return pd.DataFrame()
            
        except Exception as e:
            logger.debug(f"Real data loading failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Генерация синтетических данных для конкретного символа"""
        np.random.seed(hash(symbol) % 2**32)  # Разное семя для каждого символа
        n_points = 5000
        
        # Разные базовые цены для разных символов
        if 'BTC' in symbol.upper():
            initial_price = 45000.0
            mu = 0.0003  # Биткоин немного более волатилен
            sigma = 0.025
        elif 'ETH' in symbol.upper():
            initial_price = 3000.0
            mu = 0.0002
            sigma = 0.028  # Эфир еще более волатилен
        else:
            initial_price = 100.0
            mu = 0.0001
            sigma = 0.02
        
        # Geometric Brownian Motion для цены
        returns = np.random.normal(mu, sigma, n_points)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # OHLCV данные
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_points, freq='15min'),  # 15-минутные данные
            'open': prices * (0.999 + np.random.random(n_points) * 0.002),
            'high': prices * (1.001 + np.random.random(n_points) * 0.003),
            'low': prices * (0.997 + np.random.random(n_points) * 0.003),
            'close': prices,
            'volume': np.random.lognormal(8, 1.2, n_points)  # Разный объем для разных монет
        })
        
        data.set_index('timestamp', inplace=True)
        
        # Технические индикаторы
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['rsi'] = self._calculate_rsi(data['close'], 14)
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        # Добавляем информацию о символе
        data['symbol'] = symbol
        data['symbol_encoded'] = self.symbols.index(symbol)
        
        # Межсимвольная корреляция (BTC и ETH коррелируют)
        if len(self.symbols) > 1 and symbol != self.symbols[0]:
            # Добавляем небольшую корреляцию с первым символом
            correlation_factor = 0.7
            base_returns = returns
            correlated_noise = np.random.normal(0, sigma * 0.3, n_points)
            data['close'] *= np.exp(np.cumsum(base_returns * correlation_factor + correlated_noise))
        
        return data.fillna(0)  # Заполняем NaN нулями
    
    def _add_simple_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление простых технических индикаторов без внешних библиотек"""
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
        df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        
        # RSI (простая версия)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(20, min_periods=1).mean()
        rolling_std = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(20, min_periods=1).std()
        
        return df
    
    def _create_minimal_data(self, symbol: str) -> pd.DataFrame:
        """Создание минимальных тестовых данных"""
        n_points = 1000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='15min')
        
        # Простые данные без сложных вычислений
        base_price = 50000.0 if 'BTC' in symbol else 3000.0
        prices = base_price + np.random.normal(0, base_price * 0.01, n_points)
        
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(1000, 10000, n_points),
            'symbol': symbol,
            'symbol_encoded': self.symbols.index(symbol) if hasattr(self, 'symbols') else 0
        }, index=dates)
        
        # Добавляем минимальные технические индикаторы
        data['sma_20'] = data['close'].rolling(20, min_periods=1).mean()
        data['rsi'] = 50.0  # Нейтральное значение
        data['volatility'] = 0.02  # Фиксированное значение
        
        return data.fillna(0)
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Генерация синтетических данных (backward compatibility)"""
        return self._generate_synthetic_data_for_symbol(self.symbols[0] if self.symbols else 'BTCUSDT')
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Простой расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _add_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление features микроструктуры рынка"""
        df = data.copy()
        
        # Спреды и impact
        df['bid_ask_spread'] = (df['high'] - df['low']) / df['close']
        df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(10).sum() / df['volume'].rolling(10).sum()
        
        # Ликвидность
        df['amihud_illiquidity'] = abs(df['close'].pct_change()) / (df['volume'] + 1e-8)
        df['volume_imbalance'] = df['volume'].rolling(5).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
        
        # Momentum и mean reversion
        df['price_momentum'] = df['close'].pct_change(5)
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Volatility clustering
        returns = df['close'].pct_change()
        df['garch_vol'] = returns.rolling(20).std()
        df['vol_of_vol'] = df['garch_vol'].rolling(5).std()
        
        # Intraday patterns (если есть timestamp)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour / 24.0
            df['day_of_week'] = df.index.dayofweek / 7.0
        
        return df
    
    def _setup_spaces(self):
        """Настройка пространств наблюдений и действий"""
        # Observation space: lookback window + portfolio state + market microstructure + symbol info
        # Подсчитываем только числовые колонки для правильного размера observation space
        sample_data = self.data.select_dtypes(include=[np.number])
        n_market_features = len(sample_data.columns)
        n_portfolio_features = 7  # Из _get_portfolio_features
        n_microstructure_features = 5  # Из _get_microstructure_features
        n_features = n_market_features + n_portfolio_features + n_microstructure_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, n_features),
            dtype=np.float32
        )
        
        # Action space: более сложное пространство действий
        # [trade_signal, position_size, order_type, stop_loss, take_profit]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 0.1, 0.1]),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Сброс среды"""
        super().reset(seed=seed)
        
        # Сброс состояния
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        
        # Сброс позиций для всех символов
        for symbol in self.symbols:
            self.positions[symbol] = Position(symbol, 0.0, 0.0, 0.0, 0.0, 0)
        
        # Начинаем с первого символа
        self.current_symbol_index = 0
        self.current_symbol = self.symbols[self.current_symbol_index]
        self.data = self.multi_symbol_data[self.current_symbol]
        self.position = self.positions[self.current_symbol]
        self.steps_since_rotation = 0
        
        # Очистка истории
        self.trades_history.clear()
        self.portfolio_history.clear()
        
        # Сброс метрик
        self.reset_metrics()
        
        logger.info(f"🔄 Среда сброшена. Начинаем с символа: {self.current_symbol}")
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение шага с чередованием символов"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Чередование символов каждые N шагов
        self.steps_since_rotation += 1
        if self.steps_since_rotation >= self.symbol_rotation_interval:
            self._rotate_symbol()
            self.steps_since_rotation = 0
            
        # Извлечение действий
        trade_signal = float(action[0])  # -1 to 1 (sell to buy)
        position_size = float(action[1])  # 0 to 1 (fraction of portfolio)
        order_type_raw = float(action[2])  # 0 to 1 (market to limit)
        stop_loss = float(action[3])  # 0 to 0.1 (percentage)
        take_profit = float(action[4])  # 0 to 0.1 (percentage)
        
        # Получение текущих рыночных данных
        current_data = self.data.iloc[self.current_step]
        current_price = current_data['close']
        current_volume = current_data['volume']
        
        # Сохранение предыдущей стоимости портфеля
        previous_portfolio_value = self._get_portfolio_value()
        
        # Исполнение торговых действий
        trade_info = self._execute_trade(
            trade_signal, position_size, current_price, 
            current_volume, current_data
        )
        
        # Обновление состояния
        self.current_step += 1
        self._update_position(current_price)
        
        # Расчет текущей стоимости портфеля
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(current_portfolio_value)
        
        # Расчет вознаграждения
        market_return = current_data.get('close', 0) / self.data.iloc[self.current_step-1].get('close', 1) - 1
        volatility = current_data.get('volatility', 0.02)
        
        reward = self.reward_scheme.calculate_reward(
            current_portfolio_value=current_portfolio_value,
            previous_portfolio_value=previous_portfolio_value,
            current_position=self.position,
            market_return=market_return,
            volatility=volatility,
            trade_info=trade_info
        )
        
        # Проверка завершения - используем более мягкие условия
        terminated = self.current_step >= min(len(self.data) - 10, self.max_steps)  # Оставляем запас данных
        
        # Проверка критических потерь - более мягкие условия для обучения
        total_loss = (current_portfolio_value - self.initial_balance) / self.initial_balance
        truncated = total_loss < -0.8 or current_portfolio_value < self.initial_balance * 0.1  # Остановка при потере 80% или портфель < 10%
        
        # Убеждаемся что terminated и truncated - булевы значения
        terminated = bool(terminated)
        truncated = bool(truncated)
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _rotate_symbol(self):
        """Переключение на следующий символ в списке"""
        # Сохраняем текущую позицию
        self.positions[self.current_symbol] = self.position
        
        # Переключаемся на следующий символ
        self.current_symbol_index = (self.current_symbol_index + 1) % len(self.symbols)
        self.current_symbol = self.symbols[self.current_symbol_index]
        
        # Обновляем данные и позицию
        self.data = self.multi_symbol_data[self.current_symbol]
        self.position = self.positions[self.current_symbol]
        
        # Синхронизируем current_step с новыми данными
        # Это важно, чтобы не выйти за границы данных нового символа
        self.current_step = min(self.current_step, len(self.data) - 1)
        
        logger.debug(f"🔄 Переключились на {self.current_symbol}, "
                    f"позиция: {self.position.size:.6f}, "
                    f"шаг: {self.current_step}")
    
    def _execute_trade(self, 
                      trade_signal: float, 
                      position_size: float,
                      current_price: float,
                      current_volume: float,
                      market_data: pd.Series) -> Optional[Trade]:
        """Реалистичное исполнение торговых операций"""
        
        # Определение направления и размера торговли
        if abs(trade_signal) < 0.1:  # Малые сигналы игнорируются
            return None
            
        # Расчет размера ордера
        portfolio_value = self._get_portfolio_value()
        max_trade_value = portfolio_value * position_size
        
        if trade_signal > 0:  # BUY
            if max_trade_value < self.min_trade_size:
                return None
                
            # Проверка достаточности баланса
            if max_trade_value > self.balance:
                max_trade_value = self.balance * 0.95  # Оставляем 5% резерва
                
            if max_trade_value < self.min_trade_size:
                return None
                
            order_side = OrderSide.BUY
            trade_value = max_trade_value
            
        else:  # SELL
            if self.position.size <= 0:
                return None
                
            # Размер продажи ограничен текущей позицией
            max_sell_size = self.position.size * position_size
            trade_value = max_sell_size * current_price
            
            if trade_value < self.min_trade_size:
                return None
                
            order_side = OrderSide.SELL
        
        # Продвинутое моделирование исполнения ордера
        executed_price, slippage, market_impact, execution_time, is_full_fill = self._simulate_order_execution(
            order_side=order_side,
            trade_value=trade_value,
            current_price=current_price,
            current_volume=current_volume,
            volatility=market_data.get('volatility', 0.02)
        )
        
        # Если частичное исполнение, корректируем размер сделки
        if not is_full_fill:
            order_size_in_units = trade_value / current_price
            available_liquidity = current_volume * 0.1  # Примерная доступная ликвидность
            filled_size, _ = self.liquidity_model.simulate_partial_fill(
                order_size_in_units, available_liquidity, market_data.get('volatility', 0.02)
            )
            trade_value = filled_size * executed_price
        
        # Расчет количества и комиссии
        if order_side == OrderSide.BUY:
            quantity = trade_value / executed_price
            commission = trade_value * self.commission_rate
            
            # Обновление баланса и позиции
            total_cost = trade_value + commission
            if total_cost <= self.balance:
                self.balance -= total_cost
                
                # Обновление позиции (weighted average price)
                if self.position.size > 0:
                    total_size = self.position.size + quantity
                    total_value = (self.position.size * self.position.avg_price) + (quantity * executed_price)
                    self.position.avg_price = total_value / total_size
                    self.position.size = total_size
                else:
                    self.position.size = quantity
                    self.position.avg_price = executed_price
                    
                self.position.last_update = self.current_step
                
        else:  # SELL
            quantity = min(trade_value / executed_price, self.position.size)
            revenue = quantity * executed_price
            commission = revenue * self.commission_rate
            
            # Расчет realized PnL
            realized_pnl = (executed_price - self.position.avg_price) * quantity
            
            # Обновление баланса и позиции
            self.balance += (revenue - commission)
            self.position.size -= quantity
            self.position.realized_pnl += realized_pnl
            
            if self.position.size <= 1e-8:  # Практически закрыли позицию
                self.position.size = 0.0
                self.position.avg_price = 0.0
        
        # Создание записи о сделке с дополнительной информацией
        trade_info = Trade(
            timestamp=self.current_step,
            price=executed_price,
            quantity=quantity,
            side=order_side,
            order_type=OrderType.MARKET,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            realized_pnl=realized_pnl if order_side == OrderSide.SELL else 0.0
        )
        
        # Добавляем дополнительную информацию о реализме исполнения
        trade_info.execution_time = execution_time
        trade_info.is_full_fill = is_full_fill
        
        logger.debug(f"Исполнена сделка: {order_side.value} {quantity:.6f} по цене {executed_price:.6f}, "
                    f"slippage: {slippage:.4f}, impact: {market_impact:.4f}, "
                    f"время: {execution_time}, полное исполнение: {is_full_fill}")
        
        self.trades_history.append(trade_info)
        return trade_info
    
    def _simulate_order_execution(self,
                                order_side: OrderSide,
                                trade_value: float,
                                current_price: float,
                                current_volume: float,
                                volatility: float) -> Tuple[float, float, float, int, bool]:
        """
        Продвинутое симулирование реалистичного исполнения ордера
        Возвращает: (executed_price, slippage, market_impact, execution_time, is_full_fill)
        """
        
        # Получаем текущее время суток для реалистичного моделирования
        time_of_day = (self.current_step % (24 * 60)) / (24 * 60)  # Нормализованное время суток
        
        # Генерируем реалистичную книгу ордеров
        bids, asks = self.liquidity_model.generate_realistic_order_book(
            current_price, current_volume, volatility, time_of_day
        )
        
        # Определяем доступную ликвидность
        if order_side == OrderSide.BUY:
            available_liquidity = sum(level.quantity for level in asks[:5])  # Топ 5 уровней
            bid_ask_spread = asks[0].price - bids[0].price if asks and bids else current_price * 0.001
        else:
            available_liquidity = sum(level.quantity for level in bids[:5])
            bid_ask_spread = asks[0].price - bids[0].price if asks and bids else current_price * 0.001
        
        # Рассчитываем рыночное воздействие и время исполнения
        market_impact, execution_time = self.liquidity_model.calculate_realistic_market_impact(
            order_size=trade_value,
            current_volume=current_volume,
            volatility=volatility,
            bid_ask_spread=bid_ask_spread
        )
        
        # Симулируем частичное исполнение
        order_size_in_units = trade_value / current_price
        filled_size, is_full_fill = self.liquidity_model.simulate_partial_fill(
            order_size_in_units, available_liquidity, volatility
        )
        
        # Базовое проскальзывание с векторизацией
        base_slippage = self.config.get('base_slippage', 0.0005)
        
        if self.enable_slippage:
            # Используем векторизованную функцию для расчета slippage
            order_sizes = np.array([trade_value])
            volumes = np.array([current_volume])
            volatilities = np.array([volatility])
            
            slippage = calculate_slippage_vectorized(order_sizes, volumes, volatilities, base_slippage)[0]
            
            # Добавляем влияние времени суток (больше slippage в нетрадиционные часы)
            time_factor = 1.0 + 0.2 * abs(time_of_day - 0.5)
            slippage *= time_factor
            
            # Добавляем стохастический компонент
            slippage *= (1 + np.random.normal(0, 0.15))
            slippage = np.clip(slippage, 0, 0.02)  # Максимум 2%
        else:
            slippage = 0.0
        
        # Расчет окончательной цены исполнения
        if order_side == OrderSide.BUY:
            # При покупке цена хуже (выше)
            executed_price = current_price * (1 + slippage + market_impact)
            
            # Если частичное исполнение, цена может быть лучше
            if not is_full_fill:
                executed_price *= 0.998  # Небольшое улучшение цены
        else:
            # При продаже цена хуже (ниже)
            executed_price = current_price * (1 - slippage - market_impact)
            
            # Если частичное исполнение, цена может быть лучше
            if not is_full_fill:
                executed_price *= 1.002  # Небольшое улучшение цены
        
        # Корректируем размер сделки для частичного исполнения
        actual_trade_value = trade_value * (filled_size / order_size_in_units) if not is_full_fill else trade_value
        
        return executed_price, slippage, market_impact, execution_time, is_full_fill
    
    def _update_position(self, current_price: float):
        """Обновление нереализованного PnL позиции"""
        if self.position.size > 0:
            self.position.unrealized_pnl = (current_price - self.position.avg_price) * self.position.size
        else:
            self.position.unrealized_pnl = 0.0
    
    def _get_portfolio_value(self) -> float:
        """Получение текущей стоимости портфеля (все символы)"""
        total_value = self.balance
        
        # Добавляем стоимость позиций по всем символам
        for symbol, position in self.positions.items():
            if position.size > 0:
                # Получаем текущую цену для каждого символа
                symbol_data = self.multi_symbol_data[symbol]
                if self.current_step < len(symbol_data):
                    current_price = symbol_data.iloc[min(self.current_step, len(symbol_data)-1)]['close']
                    position_value = position.size * current_price
                    total_value += position_value
        
        return total_value
    
    def _get_observation(self) -> np.ndarray:
        """
        Создание observation с улучшенной нормализацией
        Основано на лучших практиках из FinRL
        """
        # Получение окна данных
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx].copy()
        
        # Дополнение данных если окно меньше требуемого
        if len(window_data) < self.lookback_window:
            padding = self.lookback_window - len(window_data)
            last_row = window_data.iloc[-1] if len(window_data) > 0 else self.data.iloc[0]
            padding_data = pd.DataFrame([last_row] * padding, columns=self.data.columns)
            window_data = pd.concat([padding_data, window_data])
        
        # Продвинутая нормализация (как в FinRL)
        normalized_data = self._normalize_market_data(window_data)
        
        # Фильтруем только числовые колонки
        numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
        normalized_data_numeric = normalized_data[numeric_cols]
        
        # Добавление portfolio state
        portfolio_features = self._get_portfolio_features()
        
        # Добавление microstructure features
        microstructure_features = self._get_microstructure_features()
        
        # Объединение всех features
        portfolio_matrix = np.tile(portfolio_features, (self.lookback_window, 1))
        microstructure_matrix = np.tile(microstructure_features, (self.lookback_window, 1))
        
        observation = np.concatenate([
            normalized_data_numeric.values,
            portfolio_matrix,
            microstructure_matrix
        ], axis=1).astype(np.float32)
        
        # Финальная проверка на NaN и Inf
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Клампинг для стабильности
        observation = np.clip(observation, -10.0, 10.0)
        
        return observation
    
    def _normalize_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Продвинутая нормализация рыночных данных"""
        normalized = data.copy()
        
        # Цены нормализуются как относительные изменения
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in normalized.columns:
                normalized[col] = normalized[col].pct_change().fillna(0)
                normalized[col] = np.clip(normalized[col], -0.1, 0.1) * 10  # Масштабирование
        
        # Объем логарифмически нормализуется
        if 'volume' in normalized.columns:
            volume_mean = normalized['volume'].mean()
            if volume_mean > 0:
                normalized['volume'] = np.log1p(normalized['volume'] / volume_mean)
                normalized['volume'] = np.clip(normalized['volume'], -3, 3)
        
        # Технические индикаторы нормализуются индивидуально
        # Исключаем текстовые колонки
        exclude_cols = price_cols + ['volume', 'symbol', 'symbol_encoded']
        for col in normalized.columns:
            if col not in exclude_cols:
                try:
                    col_std = normalized[col].std()
                    col_mean = normalized[col].mean()
                    
                    if col_std > 0:
                        normalized[col] = (normalized[col] - col_mean) / col_std
                        normalized[col] = np.clip(normalized[col], -3, 3)
                    else:
                        normalized[col] = 0
                except (TypeError, ValueError):
                    # Пропускаем колонки, которые нельзя нормализовать
                    normalized[col] = 0
        
        return normalized.fillna(0)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Получение нормализованных features портфеля с мультиактивной информацией"""
        portfolio_value = self._get_portfolio_value()
        current_price = self.data.iloc[self.current_step]['close']
        
        features = np.array([
            # Баланс как доля от начального капитала
            self.balance / self.initial_balance,
            
            # Размер текущей позиции как доля от портфеля
            (self.position.size * current_price) / portfolio_value if portfolio_value > 0 else 0,
            
            # Общая доходность
            (portfolio_value - self.initial_balance) / self.initial_balance,
            
            # Нереализованный PnL текущей позиции как доля от портфеля
            self.position.unrealized_pnl / portfolio_value if portfolio_value > 0 else 0,
            
            # Реализованный PnL текущей позиции как доля от начального капитала
            self.position.realized_pnl / self.initial_balance,
            
            # Текущий символ (нормализованный индекс)
            self.current_symbol_index / max(len(self.symbols) - 1, 1),
            
            # Общее количество позиций по всем символам
            sum(1 for pos in self.positions.values() if pos.size > 0) / len(self.symbols)
        ])
        
        # Нормализация и клампинг
        features = np.tanh(features)  # Мягкая нормализация
        return features
    
    def _get_microstructure_features(self) -> np.ndarray:
        """Получение features микроструктуры рынка"""
        current_data = self.data.iloc[self.current_step]
        
        features = np.array([
            # Spread
            current_data.get('bid_ask_spread', 0.001),
            
            # Ликвидность
            current_data.get('amihud_illiquidity', 0.0),
            
            # Volume imbalance
            current_data.get('volume_imbalance', 0.0),
            
            # Momentum
            current_data.get('price_momentum', 0.0),
            
            # Mean reversion
            current_data.get('mean_reversion', 0.0)
        ])
        
        # Нормализация
        features = np.clip(features, -3, 3)
        return features
    
    def _get_info(self) -> Dict:
        """Получение информации о состоянии среды с мультиактивной информацией"""
        portfolio_value = self._get_portfolio_value()
        current_price = self.data.iloc[self.current_step]['close']
        
        # Расчет метрик
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (приблизительный)
        if len(self.portfolio_history) > 10:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Win rate
        profitable_trades = sum(1 for trade in self.trades_history if trade.realized_pnl > 0)
        total_completed_trades = sum(1 for trade in self.trades_history if trade.side == OrderSide.SELL)
        win_rate = profitable_trades / max(total_completed_trades, 1)
        
        # Информация о позициях по всем символам
        positions_info = {}
        total_position_value = 0
        for symbol, position in self.positions.items():
            if position.size > 0:
                symbol_data = self.multi_symbol_data[symbol]
                symbol_price = symbol_data.iloc[min(self.current_step, len(symbol_data)-1)]['close']
                position_value = position.size * symbol_price
                total_position_value += position_value
                
                positions_info[symbol] = {
                    'size': position.size,
                    'avg_price': position.avg_price,
                    'current_price': symbol_price,
                    'position_value': position_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl
                }
        
        return {
            # Общая информация
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': len(self.trades_history),
            'step': self.current_step,
            
            # Текущий активный символ
            'current_symbol': self.current_symbol,
            'current_symbol_index': self.current_symbol_index,
            'current_price': current_price,
            'steps_since_rotation': self.steps_since_rotation,
            
            # Текущая позиция
            'position_size': self.position.size,
            'position_value': self.position.size * current_price,
            'unrealized_pnl': self.position.unrealized_pnl,
            'realized_pnl': self.position.realized_pnl,
            
            # Информация о всех позициях
            'positions': positions_info,
            'total_position_value': total_position_value,
            'active_positions_count': sum(1 for pos in self.positions.values() if pos.size > 0),
            
            # Доходность по символам
            'symbols_info': {
                symbol: {
                    'current_price': self.multi_symbol_data[symbol].iloc[min(self.current_step, len(self.multi_symbol_data[symbol])-1)]['close'],
                    'symbol_encoded': i
                }
                for i, symbol in enumerate(self.symbols)
            }
        }
    
    def reset_metrics(self):
        """Сброс метрик производительности"""
        if hasattr(self.reward_scheme, 'reset_metrics'):
            self.reward_scheme.reset_metrics()
    
    def render(self, mode='human'):
        """Отображение состояния среды"""
        info = self._get_info()
        
        print(f"\n=== Trading Environment State ===")
        print(f"Step: {info['step']}")
        print(f"Portfolio Value: ${info['portfolio_value']:,.2f}")
        print(f"Balance: ${info['balance']:,.2f}")
        print(f"Position: {info['position_size']:.6f} @ ${self.position.avg_price:.2f}")
        print(f"Unrealized PnL: ${info['unrealized_pnl']:,.2f}")
        print(f"Total Return: {info['total_return']:.2%}")
        print(f"Sharpe Ratio: {info['sharpe_ratio']:.3f}")
        print(f"Win Rate: {info['win_rate']:.2%}")
        print(f"Total Trades: {info['total_trades']}")
        print(f"Current Price: ${info['current_price']:,.2f}")
        print("=" * 35)

# Конфигурация по умолчанию для мультиактивной торговли
DEFAULT_CONFIG = {
    'symbols': ['BTCUSDT', 'ETHUSDT'],  # Мультиактивная торговля BTC и ETH
    'symbol_rotation_interval': 1,  # Переключение каждый шаг
    'timeframe': '15m',  # 15-минутные данные
    'exchange': 'binance',
    'initial_balance': 100000.0,
    'commission_rate': 0.001,
    'min_trade_size': 10.0,
    'max_steps': 5000,
    'lookback_window': 50,
    'enable_slippage': True,
    'enable_market_impact': True,
    'enable_liquidity_modeling': True,
    'enable_order_book': True,
    'enable_partial_fills': True,
    'max_order_book_levels': 20,
    'base_slippage': 0.0005,
    'risk_tolerance': 0.02,
    'transaction_cost_penalty': 1.0,
    'stability_bonus': 0.1
}

def create_trading_environment(config: Optional[Dict[str, Any]] = None) -> RealisticTradingEnvironment:
    """Фабричная функция для создания торговой среды"""
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    return RealisticTradingEnvironment(config)

# Пример использования
if __name__ == "__main__":
    config = {
        'symbol': 'BTCUSDT',
        'initial_balance': 10000.0,
        'lookback_window': 30
    }
    
    env = create_trading_environment(config)
    
    # Тестирование среды
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Несколько случайных шагов
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward:.4f}")
        print(f"Portfolio: ${info['portfolio_value']:,.2f}")
        
        if terminated or truncated:
            break
    
    env.render()