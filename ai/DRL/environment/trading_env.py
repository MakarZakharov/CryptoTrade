import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import os
import sys

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

try:
    from ai.ML1.market_analysis.data.features.technical_indicators import TechnicalIndicators
    from ai.ML1.market_analysis.data.fetchers.csv_fetcher import CSVFetcher
    from ai.DRL.config.trading_config import TradingConfig
    from ai.DRL.environment.reward_schemes import (
        create_default_reward_scheme, create_conservative_reward_scheme, 
        create_aggressive_reward_scheme, create_optimized_reward_scheme, CompositeRewardScheme
    )
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все модули находятся в правильных директориях")
    raise



class TradingEnv(gym.Env):
    """
    Реалистичная среда для обучения DRL-агента торговле криптовалютными парами.
    Поддерживает проскальзывание, комиссии, спред, частичное исполнение и моделирование ликвидности.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config: TradingConfig):
        super(TradingEnv, self).__init__()
        self.config = config
        
        # Загрузка и подготовка данных
        self.data = self._load_data()
        if self.data.empty:
            raise ValueError(f"Данные не загружены для {config.symbol}")
        
        # Торговые параметры
        self.initial_balance = config.initial_balance
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
        self.spread_rate = config.spread_rate
        
        # Состояние среды
        self.current_step = 0
        self.balance = self.initial_balance  # USDT баланс
        self.crypto_balance = 0.0  # Количество криптовалюты
        self.total_trades = 0
        self.profitable_trades = 0
        
        # История для метрик
        self.portfolio_history = []
        self.trade_history = []
        self.drawdown_history = []
        
        # Определение пространств
        self._setup_spaces()
        
        # Инициализация схемы наград
        self._setup_reward_scheme()
        
        # Инициализация метрик
        self.reset_metrics()

    def _load_data(self) -> pd.DataFrame:
        """Загрузка данных с техническими индикаторами."""
        try:
            # Ensure we use absolute path to data directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            data_path = os.path.join(project_root, 'data', self.config.exchange)
            
            fetcher = CSVFetcher(
                symbol=self.config.symbol,
                interval=self.config.timeframe,
                base_path=data_path
            )
            
            # Загружаем данные за весь период
            data = fetcher.fetch_data(
                start_date='2018-01-01',
                end_date='2024-12-31'
            )
            
            if data.empty:
                print(f"Данные не найдены для {self.config.symbol}")
                return data
            
            # Добавляем технические индикаторы
            if self.config.include_technical_indicators:
                indicators_to_include = list(self.config.indicator_periods.keys())
                data = TechnicalIndicators.add_all_indicators(data, include=indicators_to_include)
            
            # Добавляем дополнительные фичи
            data = self._add_market_features(data)
            
            return data.dropna()
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame()

    def _add_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавление дополнительных рыночных фичей."""
        df = data.copy()
        
        # Ценовые фичи
        df['price_change'] = df['close'].pct_change()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Волатильность
        df['volatility_5'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Время (для цикличности)
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour / 24.0
            df['day_of_week'] = df.index.dayofweek / 7.0
            df['day_of_month'] = df.index.day / 31.0
            df['month'] = df.index.month / 12.0
        
        return df

    def _setup_spaces(self):
        """Настройка пространств наблюдения и действий."""
        # Определяем размер наблюдения
        lookback = self.config.lookback_window
        n_features = len(self.data.columns) + 3  # +3 для баланса, позиции, портфеля
        
        # Пространство наблюдения: окно цен + состояние портфеля
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback, n_features), 
            dtype=np.float32
        )
        
        # Пространство действий: [percentage_to_trade]
        # percentage_to_trade: от -1 (продать все) до 1 (купить на весь баланс)
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

    def _setup_reward_scheme(self):
        """Настройка схемы наград."""
        if self.config.reward_scheme == 'conservative':
            self.reward_scheme = create_conservative_reward_scheme()
        elif self.config.reward_scheme == 'aggressive':
            self.reward_scheme = create_aggressive_reward_scheme()
        elif self.config.reward_scheme == 'optimized':
            self.reward_scheme = create_optimized_reward_scheme()
        elif self.config.reward_scheme == 'custom' and self.config.custom_reward_weights:
            # Создаем кастомную схему на основе весов
            from ai.DRL.environment.reward_schemes import (
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
            # По умолчанию
            self.reward_scheme = create_default_reward_scheme()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Сброс среды к начальному состоянию."""
        super().reset(seed=seed)
        self.current_step = self.config.lookback_window
        self.balance = self.initial_balance
        self.crypto_balance = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        
        self.portfolio_history = []
        self.trade_history = []
        self.drawdown_history = []
        
        self.reset_metrics()
        
        # Сброс схемы наград
        if hasattr(self, 'reward_scheme'):
            self.reward_scheme.reset()
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнить действие и вернуть результат."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Извлекаем действие
        trade_percentage = float(action[0])
        
        # Выполняем торговое действие
        self._execute_trade(trade_percentage)
        
        # Обновляем состояние
        self.current_step += 1
        self._update_portfolio_history()
        self._update_metrics()
        
        # Рассчитываем награду через схему наград
        env_state = self._get_info()
        reward = self.reward_scheme.calculate(env_state)
        
        # Проверяем завершение эпизода
        terminated = self.current_step >= len(self.data) - 1
        truncated = self._get_portfolio_value() <= self.initial_balance * 0.1
        
        return self._get_observation(), reward, terminated, truncated, env_state

    def _execute_trade(self, trade_percentage: float) -> float:
        """Выполнение торгового действия с реалистичными условиями."""
        current_price = self.data.iloc[self.current_step]['close']
        current_volume = self.data.iloc[self.current_step]['volume']
        
        # Ограничиваем диапазон действий
        trade_percentage = np.clip(trade_percentage, -1.0, 1.0)
        
        # Применяем проскальзывание и спред
        if trade_percentage > 0:  # Покупка
            effective_price = current_price * (1 + self.slippage_rate + self.spread_rate/2)
        elif trade_percentage < 0:  # Продажа
            effective_price = current_price * (1 - self.slippage_rate - self.spread_rate/2)
        else:  # Держим
            return 0.0
        
        # Рассчитываем размер сделки - агент полностью контролирует процент
        if trade_percentage > 0:  # Покупка
            # Агент решает какую долю баланса потратить на покупку
            usdt_amount = self.balance * abs(trade_percentage)
        else:  # Продажа
            # Агент решает какую долю криптовалюты продать
            crypto_amount_to_sell = self.crypto_balance * abs(trade_percentage)
            usdt_amount = crypto_amount_to_sell * effective_price
        
        # Проверяем минимальную сумму сделки
        if abs(usdt_amount) < self.config.min_trade_amount:
            return 0.0
        
        # Моделируем влияние на ликвидность
        order_impact = self._calculate_liquidity_impact(usdt_amount, current_volume, effective_price)
        effective_price *= (1 + order_impact if trade_percentage > 0 else 1 - order_impact)
        
        # Частичное исполнение
        fill_ratio = 1.0
        if self.config.enable_partial_fills:
            fill_ratio = self._calculate_fill_ratio(usdt_amount, current_volume, effective_price)
            usdt_amount *= fill_ratio
        
        # Выполняем сделку
        old_portfolio_value = self._get_portfolio_value()
        trade_executed = False
        
        if trade_percentage > 0:  # Покупка
            if usdt_amount <= self.balance:
                commission = usdt_amount * self.commission_rate
                net_usdt = usdt_amount - commission
                crypto_amount = net_usdt / effective_price
                
                self.balance -= usdt_amount
                self.crypto_balance += crypto_amount
                self.total_trades += 1
                trade_executed = True
                
                profit = 0.0  # Прибыль пока неизвестна для покупки
                self._record_trade('buy', crypto_amount, effective_price, commission, profit)
        
        else:  # Продажа
            crypto_amount_to_sell = self.crypto_balance * abs(trade_percentage) * fill_ratio
            if crypto_amount_to_sell > 0:
                usdt_received = crypto_amount_to_sell * effective_price
                commission = usdt_received * self.commission_rate
                net_usdt = usdt_received - commission
                
                self.crypto_balance -= crypto_amount_to_sell
                self.balance += net_usdt
                self.total_trades += 1
                trade_executed = True
                
                # Рассчитываем прибыль от продажи
                avg_buy_price = self._get_average_buy_price()
                profit = (effective_price - avg_buy_price) * crypto_amount_to_sell if avg_buy_price > 0 else 0.0
                
                if profit > 0:
                    self.profitable_trades += 1
                
                self._record_trade('sell', crypto_amount_to_sell, effective_price, commission, profit)
        
        # Рассчитываем награду
        if trade_executed:
            new_portfolio_value = self._get_portfolio_value()
            reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value > 0 else 0.0
            return reward
        
        return 0.0

    def _calculate_liquidity_impact(self, usdt_amount: float, volume: float, price: float) -> float:
        """Рассчитать влияние ордера на ликвидность."""
        if volume <= 0:
            return 0.0
        
        # Размер ордера относительно объема торгов
        order_size_ratio = (usdt_amount / price) / volume
        
        # Если ордер слишком большой, увеличиваем влияние на цену
        if order_size_ratio > self.config.max_order_size_ratio:
            impact = self.config.liquidity_impact_threshold * (order_size_ratio / self.config.max_order_size_ratio)
            return min(impact, 0.01)  # Максимум 1% влияния
        
        return 0.0

    def _calculate_fill_ratio(self, usdt_amount: float, volume: float, price: float) -> float:
        """Рассчитать коэффициент частичного исполнения."""
        if volume <= 0:
            return 0.0
        
        # Размер ордера относительно объема
        order_size_ratio = (usdt_amount / price) / volume
        
        # Если ордер больше определенного процента от объема, частичное исполнение
        if order_size_ratio > self.config.max_order_size_ratio:
            fill_ratio = self.config.max_order_size_ratio / order_size_ratio
            return max(fill_ratio, 0.1)  # Минимум 10% исполнения
        
        return 1.0  # Полное исполнение

    def _record_trade(self, trade_type: str, amount: float, price: float, commission: float, profit: float = 0.0):
        """Записать сделку в историю."""
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
        """Получить среднюю цену покупки для расчета прибыли."""
        buy_trades = [trade for trade in self.trade_history if trade['type'] == 'buy']
        if not buy_trades:
            return 0.0
        
        total_amount = sum(trade['amount'] for trade in buy_trades)
        total_cost = sum(trade['amount'] * trade['price'] for trade in buy_trades)
        
        return total_cost / total_amount if total_amount > 0 else 0.0

    def _get_portfolio_value(self) -> float:
        """Получить текущую стоимость портфеля."""
        if self.current_step >= len(self.data):
            return self.balance
        
        current_price = self.data.iloc[self.current_step]['close']
        return self.balance + self.crypto_balance * current_price

    def _get_observation(self) -> np.ndarray:
        """Получить текущее наблюдение."""
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        
        # Получаем окно данных
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Если окно меньше требуемого, дополняем нулями
        if len(window_data) < self.config.lookback_window:
            padding = self.config.lookback_window - len(window_data)
            window_data = pd.concat([
                pd.DataFrame(np.zeros((padding, len(self.data.columns))), columns=self.data.columns),
                window_data
            ])
        
        # Нормализация данных
        observation = window_data.values.astype(np.float32)
        
        # Добавляем состояние портфеля
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.crypto_balance * self.data.iloc[self.current_step]['close'] / self.initial_balance,
            self._get_portfolio_value() / self.initial_balance
        ])
        
        # Расширяем состояние портфеля до размера окна
        portfolio_features = np.tile(portfolio_state, (self.config.lookback_window, 1))
        
        # Объединяем наблюдения
        full_observation = np.concatenate([observation, portfolio_features], axis=1)
        
        return full_observation

    def _update_portfolio_history(self):
        """Обновить историю портфеля."""
        portfolio_value = self._get_portfolio_value()
        self.portfolio_history.append(portfolio_value)

    def _update_metrics(self):
        """Обновить метрики производительности."""
        if len(self.portfolio_history) < 2:
            return
        
        # Рассчитываем просадку
        peak = max(self.portfolio_history)
        current_value = self.portfolio_history[-1]
        drawdown = (peak - current_value) / peak if peak > 0 else 0
        self.drawdown_history.append(drawdown)

    def reset_metrics(self):
        """Сброс метрик."""
        self.max_drawdown = 0.0
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0

    def _get_info(self) -> Dict:
        """Получить информацию о состоянии среды."""
        portfolio_value = self._get_portfolio_value()
        
        # Рассчитываем метрики
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0
        avg_drawdown = np.mean(self.drawdown_history) if self.drawdown_history else 0.0
        win_rate = self.profitable_trades / max(self.total_trades, 1)
        
        # Месячная прибыль (приблизительно)
        if len(self.portfolio_history) > 30:
            monthly_returns = []
            for i in range(30, len(self.portfolio_history), 30):
                start_val = self.portfolio_history[i-30]
                end_val = self.portfolio_history[i]
                if start_val > 0:
                    monthly_return = (end_val - start_val) / start_val
                    monthly_returns.append(monthly_return)
            avg_monthly_return = np.mean(monthly_returns) if monthly_returns else 0.0
        else:
            avg_monthly_return = 0.0
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_history': self.portfolio_history,
            'balance': self.balance,
            'crypto_balance': self.crypto_balance,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'avg_monthly_return': avg_monthly_return,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'current_price': self.data.iloc[self.current_step]['close'],
            'step': self.current_step
        }

    def render(self, mode='human'):
        """Визуализация состояния среды."""
        info = self._get_info()
        print(f"Step: {info['step']}")
        print(f"Portfolio Value: {info['portfolio_value']:.2f} USDT")
        print(f"Balance: {info['balance']:.2f} USDT")
        print(f"Crypto: {info['crypto_balance']:.6f}")
        print(f"Total Return: {info['total_return']:.2%}")
        print(f"Max Drawdown: {info['max_drawdown']:.2%}")
        print(f"Win Rate: {info['win_rate']:.2%}")
        print(f"Current Price: {info['current_price']:.2f}")
        print("-" * 40) 