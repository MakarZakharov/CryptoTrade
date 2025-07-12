import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import sys

# Додаємо шлях до модулів проекту
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

try:
    from CryptoTrade.ai.ML1.market_analysis.data.features.technical_indicators import TechnicalIndicators
    from CryptoTrade.ai.ML1.market_analysis.data.fetchers.csv_fetcher import CSVFetcher
    from CryptoTrade.ai.STAS_ML.config.trading_config import TradingConfig
    from CryptoTrade.ai.STAS_ML.environment.reward_schemes import (
        create_default_reward_scheme, create_conservative_reward_scheme, 
        create_aggressive_reward_scheme, create_optimized_reward_scheme, 
        create_bear_market_optimized_reward_scheme, create_static_reward_scheme, 
        CompositeRewardScheme
    )
except ImportError as e:
    print(f"Помилка імпорту: {e}")
    print("Переконайтеся, що всі модулі знаходяться в правильних директоріях")
    raise


class TradingEnv(gym.Env):
    """
    Реалістичне середовище для навчання STAS_ML-агента торгівлі криптовалютними парами.
    Підтримує проскальзування, комісії, спред, часткове виконання та моделювання ліквідності.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: TradingConfig):
        super(TradingEnv, self).__init__()
        self.config = config
        
        # Завантаження та підготовка даних
        self.data = self._load_data()
        if self.data.empty:
            raise ValueError(f"Дані не завантажені для {config.symbol}")
        
        # Торгові параметри
        self.initial_balance = config.initial_balance
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
        self.spread_rate = config.spread_rate
        
        # Стан середовища
        self.current_step = 0
        self.balance = self.initial_balance  # USDT баланс
        self.crypto_balance = 0.0  # Кількість криптовалюти
        self.total_trades = 0
        self.profitable_trades = 0
        
        # ВИПРАВЛЕННЯ: Додаємо кращий трекінг угод
        self.completed_trades = 0  # Кількість завершених циклів купівля-продаж
        self.profitable_completed_trades = 0  # Кількість прибуткових завершених циклів
        self.total_realized_pnl = 0.0  # Загальний реалізований P&L
        
        # Історія для метрик
        self.portfolio_history = []
        self.trade_history = []
        self.drawdown_history = []
        
        # Визначення просторів
        self._setup_spaces()
        
        # Ініціалізація схеми винагород
        self._setup_reward_scheme()
        
        # Ініціалізація метрик
        self.reset_metrics()

    def _load_data(self) -> pd.DataFrame:
        """Завантаження даних з технічними індикаторами."""
        try:
            # Забезпечуємо використання абсолютного шляху до директорії даних
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            data_path = os.path.join(project_root, 'data', self.config.exchange)
            
            fetcher = CSVFetcher(
                symbol=self.config.symbol,
                interval=self.config.timeframe,
                base_path=data_path
            )
            
            # Завантажуємо дані за весь період
            data = fetcher.fetch_data(
                start_date='2018-01-01',
                end_date='2024-12-31'
            )
            
            if data.empty:
                print(f"Дані не знайдені для {self.config.symbol}")
                return data
            
            # Додаємо технічні індикатори
            if self.config.include_technical_indicators:
                indicators_to_include = list(self.config.indicator_periods.keys())
                data = TechnicalIndicators.add_all_indicators(data, include=indicators_to_include)
            
            # Додаємо додаткові фічі
            data = self._add_market_features(data)
            
            return data.dropna()
            
        except Exception as e:
            print(f"Помилка завантаження даних: {e}")
            return pd.DataFrame()

    def _add_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Додавання додаткових ринкових фічей."""
        df = data.copy()
        
        # Цінові фічі
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Волатильність
        df['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Час (для цикличності)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour / 24.0
            df['day_of_week'] = df.index.dayofweek / 7.0
            df['day_of_month'] = df.index.day / 31.0
            df['month'] = df.index.month / 12.0
        
        return df

    def _setup_spaces(self):
        """Налаштування просторів спостереження та дій."""
        # Визначаємо розмір спостереження
        lookback = self.config.lookback_window
        n_features = len(self.data.columns) + 3  # +3 для балансу, позиції, портфеля
        
        # Простір спостереження: вікно цін + стан портфеля
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback, n_features), 
            dtype=np.float32
        )
        
        # ВИПРАВЛЕННЯ: Збільшуємо action space під PPO природний розподіл
        # PPO зі std=0.135 природно продукує дії ±0.3, тому встановлюємо більший діапазон
        # але обмежуємо виконання в _execute_trade()
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

    def _setup_reward_scheme(self):
        """Налаштування схеми винагород."""
        if self.config.reward_scheme == 'conservative':
            self.reward_scheme = create_conservative_reward_scheme()
        elif self.config.reward_scheme == 'aggressive':
            self.reward_scheme = create_aggressive_reward_scheme()
        elif self.config.reward_scheme == 'optimized':
            self.reward_scheme = create_optimized_reward_scheme()
        elif self.config.reward_scheme == 'bear_market_optimized':
            self.reward_scheme = create_bear_market_optimized_reward_scheme()
        elif self.config.reward_scheme == 'static':
            self.reward_scheme = create_static_reward_scheme(self.initial_balance)
        elif self.config.reward_scheme == 'custom' and self.config.custom_reward_weights:
            # Створюємо кастомну схему на основі ваг
            from CryptoTrade.ai.STAS_ML.environment.reward_schemes import (
                ProfitReward, DrawdownPenalty, SharpeRatioReward, 
                TradeQualityReward, VolatilityPenalty, ConsistencyReward
            )
            
            schemes = []
            weights = self.config.custom_reward_weights
            
            if 'profit' in weights:
                schemes.append(ProfitReward(weight=weights['profit']))
            if 'drawdown' in weights:
                schemes.append(DrawdownPenalty(weight=weights['drawdown']))
            if 'sharpe' in weights:
                schemes.append(SharpeRatioReward(weight=weights['sharpe']))
            if 'trade_quality' in weights:
                schemes.append(TradeQualityReward(weight=weights['trade_quality']))
            if 'volatility' in weights:
                schemes.append(VolatilityPenalty(weight=weights['volatility']))
            if 'consistency' in weights:
                schemes.append(ConsistencyReward(weight=weights['consistency']))
            
            self.reward_scheme = CompositeRewardScheme(schemes)
        else:
            # За замовчуванням
            self.reward_scheme = create_default_reward_scheme()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Скидання середовища до початкового стану."""
        super().reset(seed=seed)
        self.current_step = self.config.lookback_window
        self.balance = self.initial_balance
        self.crypto_balance = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # ВИПРАВЛЕННЯ: Скидаємо покращені метрики
        self.completed_trades = 0
        self.profitable_completed_trades = 0
        self.total_realized_pnl = 0.0
        
        self.portfolio_history = []
        self.trade_history = []
        self.drawdown_history = []
        
        self.reset_metrics()
        
        # Скидання схеми винагород
        if hasattr(self, 'reward_scheme'):
            self.reward_scheme.reset()
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Виконати дію та повернути результат."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Витягуємо дію
        trade_percentage = float(action[0])
        
        # Виконуємо торгову дію
        self._execute_trade(trade_percentage)
        
        # Оновлюємо стан
        self.current_step += 1
        self._update_portfolio_history()
        self._update_metrics()
        
        # Розраховуємо винагороду через схему винагород
        env_state = self._get_info()
        reward = self.reward_scheme.calculate(env_state)
        
        # Перевіряємо завершення епізоду
        terminated = self.current_step >= len(self.data) - 1
        truncated = self._get_portfolio_value() <= self.initial_balance * 0.1
        
        return self._get_observation(), reward, terminated, truncated, env_state

    def _execute_trade(self, trade_percentage: float) -> float:
        """Виконання торгової дії з управлінням ризиками."""
        current_price = self.data.iloc[self.current_step]['close']
        current_volume = self.data.iloc[self.current_step]['volume']
        
        # 1. МАСШТАБУВАННЯ ДІЙ - збільшено для більшої торгової активності
        trade_percentage = trade_percentage * 0.05  # Збільшено до 5% максимум для активної торгівлі
        trade_percentage = np.clip(trade_percentage, -0.05, 0.05)
        
        # 2. КОНТРОЛЬ ПРОСАДКИ - зменшуємо позиції при високій просадці
        if self.config.reduce_position_on_drawdown and len(self.drawdown_history) > 0:
            current_drawdown = self.drawdown_history[-1] if self.drawdown_history[-1] is not None else 0.0
            if current_drawdown > self.config.max_drawdown_limit * 0.5:  # При 7.5% просадці
                trade_percentage *= 0.5  # Зменшуємо розмір позицій на 50%
            elif current_drawdown > self.config.max_drawdown_limit * 0.75:  # При 11.25% просадці  
                trade_percentage *= 0.25  # Зменшуємо розмір позицій на 75%
        
        # 3. ДИНАМІЧНЕ УПРАВЛІННЯ РОЗМІРОМ ПОЗИЦІЙ
        if self.config.enable_position_sizing:
            portfolio_value = self._get_portfolio_value()
            risk_amount = portfolio_value * self.config.max_risk_per_trade
            
            if self.config.position_size_method == 'volatility_based':
                # Зменшуємо розмір при високій волатильності
                if 'volatility_20' in self.data.columns:
                    current_volatility = self.data.iloc[self.current_step]['volatility_20']
                    if current_volatility > 0.05:  # Висока волатільність
                        trade_percentage *= 0.7  # Зменшуємо на 30%
            elif self.config.position_size_method == 'kelly':
                # Простий Kelly criterion based на win rate
                if self.completed_trades > 5:  # Мінімум 5 угод для статистики
                    win_rate = self.profitable_completed_trades / self.completed_trades
                    if win_rate < 0.6:  # Якщо винрейт менше 60%
                        trade_percentage *= 0.8  # Зменшуємо розмір
        
        # 4. ПЕРЕВІРКА STOP-LOSS для існуючих позицій
        if self.config.enable_stop_loss and self.crypto_balance > 0:
            avg_buy_price = self._get_average_buy_price()
            if avg_buy_price and avg_buy_price > 0:
                price_change = (current_price - avg_buy_price) / avg_buy_price
                
                if self.config.stop_loss_type == 'percentage':
                    if price_change <= -self.config.stop_loss_percentage:
                        # Примусовий продаж через stop-loss
                        trade_percentage = -0.8  # Продаємо 80% позиції
                elif self.config.stop_loss_type == 'trailing':
                    # Trailing stop логіка (спрощена)
                    if hasattr(self, 'highest_price_since_buy'):
                        trailing_stop_price = self.highest_price_since_buy * (1 - self.config.trailing_stop_percentage)
                        if current_price <= trailing_stop_price:
                            trade_percentage = -0.8  # Продаємо 80% позиції
                    else:
                        self.highest_price_since_buy = current_price
                
                # Оновлюємо найвищу ціну для trailing stop
                if not hasattr(self, 'highest_price_since_buy'):
                    self.highest_price_since_buy = current_price
                else:
                    self.highest_price_since_buy = max(self.highest_price_since_buy, current_price)
        
        # 5. ІГНОРУЄМО ДУЖЕ МАЛІ ДІЇ
        if abs(trade_percentage) < 0.01:  # Збільшено поріг до 1%
            return 0.0
        
        # Застосовуємо проскальзування та спред
        if trade_percentage > 0:  # Покупка
            effective_price = current_price * (1 + self.slippage_rate + self.spread_rate/2)
        elif trade_percentage < 0:  # Продаж
            effective_price = current_price * (1 - self.slippage_rate - self.spread_rate/2)
        else:  # Тримаємо
            return 0.0
        
        # Розраховуємо розмір угоди
        if trade_percentage > 0:  # Покупка
            usdt_amount = self.balance * abs(trade_percentage)
        else:  # Продаж
            crypto_amount_to_sell = self.crypto_balance * abs(trade_percentage)
            usdt_amount = crypto_amount_to_sell * effective_price
        
        # Перевіряємо мінімальну суму угоди
        if abs(usdt_amount) < self.config.min_trade_amount:
            return 0.0
        
        # Виконуємо угоду
        trade_executed = False
        
        if trade_percentage > 0:  # Покупка
            if usdt_amount <= self.balance:
                commission = usdt_amount * self.commission_rate
                net_usdt = usdt_amount - commission
                crypto_amount = net_usdt / effective_price
                
                self.balance -= usdt_amount
                self.crypto_balance += crypto_amount
                # ВИПРАВЛЕННЯ: НЕ рахуємо покупки як окремі угоди
                # self.total_trades += 1  # Закоментовано - покупка це не завершена угода
                trade_executed = True
                
                self._record_trade('buy', crypto_amount, effective_price, commission, 0.0)
        
        else:  # Продаж
            crypto_amount_to_sell = self.crypto_balance * abs(trade_percentage)
            if crypto_amount_to_sell > 0:
                usdt_received = crypto_amount_to_sell * effective_price
                commission = usdt_received * self.commission_rate
                net_usdt = usdt_received - commission
                
                # ВИПРАВЛЕННЯ: Розраховуємо прибуток ПЕРЕД зміною балансу
                avg_buy_price = self._get_average_buy_price()
                profit = (effective_price - avg_buy_price) * crypto_amount_to_sell if avg_buy_price is not None and avg_buy_price > 0 else 0.0
                
                # Оновлюємо баланси
                self.crypto_balance -= crypto_amount_to_sell
                self.balance += net_usdt
                trade_executed = True
                
                # ВИПРАВЛЕННЯ: Правильний підрахунок угод
                # Тільки продаж вважається завершеною угодою
                self.completed_trades += 1
                self.total_trades = self.completed_trades  # Синхронізуємо для сумісності
                self.total_realized_pnl += profit
                
                if profit > 0:
                    self.profitable_trades += 1
                    self.profitable_completed_trades += 1
                
                self._record_trade('sell', crypto_amount_to_sell, effective_price, commission, profit)
        
        return 0.0

    def _record_trade(self, trade_type: str, amount: float, price: float, commission: float, profit: float = 0.0):
        """Записати угоду в історію."""
        self.trade_history.append({
            'step': self.current_step,
            'type': trade_type,
            'amount': amount,
            'price': price,
            'commission': commission,
            'profit': profit,
            'timestamp': self.data.index[self.current_step] if hasattr(self.data.index, '__getitem__') else self.current_step
        })

    def _get_average_buy_price(self) -> float:
        """Отримати середню ціну покупки для розрахунку прибутку."""
        buy_trades = [trade for trade in self.trade_history if trade['type'] == 'buy']
        if not buy_trades:
            return 0.0
        
        total_amount = sum(trade['amount'] for trade in buy_trades)
        total_cost = sum(trade['amount'] * trade['price'] for trade in buy_trades)
        
        return total_cost / total_amount if total_amount > 0 else 0.0

    def _get_portfolio_value(self) -> float:
        """Отримати поточну вартість портфеля - ВИПРАВЛЕНО з безпечною обробкою None."""
        if self.current_step >= len(self.data):
            return self.balance if self.balance is not None else self.initial_balance
        
        current_price = self.data.iloc[self.current_step]['close']
        # ВИПРАВЛЕННЯ: Безпечна обробка None значень
        balance_safe = self.balance if self.balance is not None else 0.0
        crypto_balance_safe = self.crypto_balance if self.crypto_balance is not None else 0.0
        current_price_safe = current_price if current_price is not None else 0.0
        
        portfolio_value = balance_safe + crypto_balance_safe * current_price_safe
        return portfolio_value if portfolio_value is not None else self.initial_balance

    def _get_observation(self) -> np.ndarray:
        """Отримати поточне спостереження з proper нормалізацією."""
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        
        # Отримуємо вікно даних
        window_data = self.data.iloc[start_idx:end_idx].copy()
        
        # Якщо вікно менше потрібного, доповнюємо останніми значеннями
        if len(window_data) < self.config.lookback_window:
            padding = self.config.lookback_window - len(window_data)
            last_row = window_data.iloc[-1] if len(window_data) > 0 else self.data.iloc[0]
            padding_data = pd.DataFrame([last_row] * padding, columns=self.data.columns)
            window_data = pd.concat([padding_data, window_data])
        
        # КРИТИЧНО ПОКРАЩЕНА НОРМАЛІЗАЦІЯ - виправляє нестабільність
        normalized_data = window_data.copy()
        
        # Нормалізуємо цінові колонки відносно поточної ціни (консервативно)
        current_price = self.data.iloc[self.current_step]['close']
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in normalized_data.columns:
                # Використовуємо відносні зміни замість абсолютних значень
                normalized_data[col] = (normalized_data[col] / current_price - 1.0) * 10.0
                # Жорстко обмежуємо до ±2 для стабільності
                normalized_data[col] = np.clip(normalized_data[col], -2.0, 2.0)
        
        # Консервативна нормалізація об'єму
        if 'volume' in normalized_data.columns:
            volume_mean = normalized_data['volume'].mean()
            if volume_mean > 0:
                normalized_data['volume'] = np.log1p(normalized_data['volume'] / volume_mean) / 5.0
            normalized_data['volume'] = np.clip(normalized_data['volume'], -1.0, 1.0)
        
        # ЖОРСТКО обмежуємо всі інші фічі до ±1
        for col in normalized_data.columns:
            if col not in price_columns + ['volume']:
                normalized_data[col] = np.clip(normalized_data[col], -1.0, 1.0)
        
        observation = normalized_data.values.astype(np.float32)
        
        # КОНСЕРВАТИВНО нормалізований стан портфеля
        portfolio_value = self._get_portfolio_value()
        
        # Дуже обмежена нормалізація портфеля для стабільності
        balance_ratio = min(max(self.balance / self.initial_balance, 0.1), 10.0)  # 0.1x to 10x
        crypto_value_ratio = min(max((self.crypto_balance * current_price) / self.initial_balance, 0.0), 10.0)
        portfolio_ratio = min(max(portfolio_value / self.initial_balance, 0.1), 10.0)
        
        portfolio_state = np.array([
            np.tanh((balance_ratio - 1) * 0.5),     # Дуже м'яка нормалізація балансу
            np.tanh(crypto_value_ratio * 0.5),      # Дуже м'яка нормалізація криптопозиції  
            np.tanh((portfolio_ratio - 1) * 0.5)    # Дуже м'яка нормалізація загальної вартості
        ])
        
        # Розширюємо стан портфеля до розміру вікна
        portfolio_features = np.tile(portfolio_state, (self.config.lookback_window, 1))
        
        # Об'єднуємо спостереження
        full_observation = np.concatenate([observation, portfolio_features], axis=1)
        
        # ЖОРСТКЕ фінальне обрізання для максимальної стабільності
        full_observation = np.clip(full_observation, -3.0, 3.0)
        
        return full_observation

    def _update_portfolio_history(self):
        """Оновити історію портфеля."""
        portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(portfolio_value)

    def _update_metrics(self):
        """Оновити метрики продуктивності - ВИПРАВЛЕНО."""
        if len(self.portfolio_history) < 2:
            return
        
        # ВИПРАВЛЕННЯ: Правильний розрахунок просадки
        # Просадка має розраховуватися від попереднього максимуму, а не глобального
        if not self.drawdown_history:
            # Перший запис - просадка 0
            self.drawdown_history.append(0.0)
            return
            
        # ВИПРАВЛЕННЯ: Безпечне знаходження максимуму з перевіркою на None
        current_value = self.portfolio_history[-1]
        # Фільтруємо None значення перед використанням max()
        valid_history = [v for v in self.portfolio_history if v is not None]
        peak_value = max(valid_history) if valid_history else self.initial_balance
        
        # ВИПРАВЛЕННЯ: Безпечний розрахунок просадки з перевіркою на None
        if peak_value is not None and current_value is not None and peak_value > 0:
            current_drawdown = (peak_value - current_value) / peak_value
            current_drawdown = max(0.0, current_drawdown)  # Просадка не може бути негативною
        else:
            current_drawdown = 0.0
            
        self.drawdown_history.append(current_drawdown)

    def reset_metrics(self):
        """Скидання метрик."""
        self.max_drawdown = 0.0
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0

    def _get_info(self) -> Dict:
        """Отримати інформацію про стан середовища - ВИПРАВЛЕНО."""
        portfolio_value = self._get_portfolio_value()
        
        # ВИПРАВЛЕННЯ: Правильні розрахунки метрик з безпечною обробкою None
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        # Фільтруємо None значення з drawdown_history
        valid_drawdowns = [d for d in self.drawdown_history if d is not None]
        max_drawdown = max(valid_drawdowns) if valid_drawdowns else 0.0
        avg_drawdown = np.mean(valid_drawdowns) if valid_drawdowns else 0.0
        
        # ВИПРАВЛЕННЯ: Win rate на основі завершених угод з безпечною обробкою None
        # Win rate має базуватися тільки на completed sell trades
        completed_trades_safe = self.completed_trades if self.completed_trades is not None else 0
        profitable_trades_safe = self.profitable_completed_trades if self.profitable_completed_trades is not None else 0
        win_rate = profitable_trades_safe / max(completed_trades_safe, 1)
        
        # ВИПРАВЛЕННЯ: Додаємо більше корисних метрик
        total_realized_return = self.total_realized_pnl / self.initial_balance
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_history': self.portfolio_history,
            'balance': self.balance,
            'crypto_balance': self.crypto_balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'total_trades': self.total_trades,
            'completed_trades': self.completed_trades,
            'win_rate': win_rate,
            'profitable_trades': self.profitable_trades,
            'profitable_completed_trades': self.profitable_completed_trades,
            'total_realized_pnl': self.total_realized_pnl,
            'total_realized_return': total_realized_return,
            'current_price': self.data.iloc[self.current_step]['close'],
            'step': self.current_step,
            'initial_balance': self.initial_balance  # ВИПРАВЛЕННЯ: додаємо статичний початковий баланс
        }

    def render(self, mode='human'):
        """Візуалізація стану середовища."""
        info = self._get_info()
        print(f"Крок: {info['step']}")
        print(f"Вартість портфеля: {info['portfolio_value']:.2f} USDT")
        print(f"Баланс: {info['balance']:.2f} USDT")
        print(f"Криптовалюта: {info['crypto_balance']:.6f}")
        print(f"Загальна доходність: {info['total_return']:.2%}")
        print(f"Макс. просадка: {info['max_drawdown']:.2%}")
        print(f"Винрейт: {info['win_rate']:.2%}")
        print(f"Поточна ціна: {info['current_price']:.2f}")
        print("-" * 40)