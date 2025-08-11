"""
Gymnasium-совместимое торговое окружение для DRL агентов.
Поддерживает дискретные и непрерывные действия, параметризуемые награды, векторизацию.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import warnings

from .data_loader import DataLoader
from .simulator import MarketSimulator, OrderSide, OrderType, SlippageModel
from .metrics import MetricsCalculator, PerformanceMetrics, TradeMetrics


class ActionSpace(Enum):
    """Тип action space."""
    DISCRETE = "discrete"  # {0: Hold, 1: Buy, 2: Sell}
    CONTINUOUS = "continuous"  # [-1, 1] - направление и размер позиции


class RewardType(Enum):
    """Тип функции награды."""
    PNL = "pnl"  # Чистая прибыль/убыток
    LOG_RETURN = "log_return"  # Log returns
    SHARPE = "sharpe"  # Sharpe-подобная метрика
    RISK_ADJUSTED = "risk_adjusted"  # PnL - λ·turnover - μ·drawdown
    SORTINO = "sortino"  # Sortino-подобная метрика


class CryptoTradingEnv(gym.Env):
    """
    Торговое окружение для криптовалют, совместимое с Gymnasium и Stable-Baselines3.

    Особенности:
    - Поддержка дискретных и непрерывных действий
    - Реалистичная симуляция рынка (проскальзывание, спред, комиссии)
    - Параметризуемая функция награды
    - Технические индикаторы
    - Подробное логирование
    - Train/val/test splits
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        # Данные
        data_path: Optional[str] = None,
        symbol: str = "BTCUSDT",
        timeframe: str = "1d",
        start_index: int = 0,
        end_index: Optional[int] = None,

        # Торговые параметры
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,  # Максимальный размер позиции (доля баланса)

        # Action space
        action_type: ActionSpace = ActionSpace.DISCRETE,

        # Observation
        observation_window: int = 50,  # Окно для price history
        add_indicators: bool = True,
        normalize_observations: bool = True,

        # Reward
        reward_type: RewardType = RewardType.RISK_ADJUSTED,
        reward_scaling: float = 1.0,
        turnover_penalty: float = 0.0001,  # λ для risk-adjusted
        drawdown_penalty: float = 0.001,   # μ для risk-adjusted

        # Market simulation
        maker_fee: float = 0.0001,
        taker_fee: float = 0.001,
        slippage_model: SlippageModel = SlippageModel.PERCENTAGE,
        slippage_percentage: float = 0.0005,
        spread_base: float = 0.0001,

        # Episode termination
        max_steps: Optional[int] = None,
        stop_on_bankruptcy: bool = True,
        bankruptcy_threshold: float = 0.1,  # 10% от начального баланса
        max_drawdown_threshold: Optional[float] = None,  # Например, -0.5 = -50%

        # Другое
        render_mode: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """
        Инициализация торгового окружения.

        Args:
            data_path: Путь к данным (CSV/Parquet)
            symbol: Торговая пара
            timeframe: Таймфрейм
            start_index: Начальный индекс данных
            end_index: Конечный индекс данных
            initial_balance: Начальный баланс
            max_position_size: Максимальный размер позиции (доля)
            action_type: Тип action space
            observation_window: Размер окна наблюдений
            add_indicators: Добавлять ли технические индикаторы
            normalize_observations: Нормализовать ли наблюдения
            reward_type: Тип функции награды
            reward_scaling: Масштабирование награды
            turnover_penalty: Штраф за оборот (для risk-adjusted)
            drawdown_penalty: Штраф за просадку (для risk-adjusted)
            maker_fee: Комиссия мейкера
            taker_fee: Комиссия тейкера
            slippage_model: Модель проскальзывания
            slippage_percentage: Процент проскальзывания
            spread_base: Базовый спред
            max_steps: Максимальное количество шагов в эпизоде
            stop_on_bankruptcy: Останавливать при банкротстве
            bankruptcy_threshold: Порог банкротства
            max_drawdown_threshold: Максимальная допустимая просадка
            render_mode: Режим рендеринга
            random_seed: Seed для воспроизводимости
        """
        super().__init__()

        # Параметры
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.action_type = action_type
        self.observation_window = observation_window
        self.reward_type = reward_type
        self.reward_scaling = reward_scaling
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty
        self.max_steps = max_steps
        self.stop_on_bankruptcy = stop_on_bankruptcy
        self.bankruptcy_threshold = bankruptcy_threshold * initial_balance
        self.max_drawdown_threshold = max_drawdown_threshold
        self.render_mode = render_mode

        # Загрузчик данных
        self.data_loader = DataLoader(
            data_path=data_path,
            symbol=symbol,
            timeframe=timeframe,
            normalize=normalize_observations,
            add_indicators=add_indicators
        )
        self.data_loader.load(start_index=start_index, end_index=end_index)

        if len(self.data_loader) == 0:
            raise ValueError("No data loaded!")

        # Симулятор рынка
        self.market_simulator = MarketSimulator(
            maker_fee=maker_fee,
            taker_fee=taker_fee,
            slippage_model=slippage_model,
            slippage_percentage=slippage_percentage,
            spread_base=spread_base,
            random_seed=random_seed
        )

        # Калькулятор метрик
        self.metrics_calculator = MetricsCalculator()

        # Определяем spaces
        self._setup_spaces()

        # State переменные
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.equity_curve: List[float] = []
        self.balance_history: List[float] = []
        self.trades_history: List[TradeMetrics] = []
        self.current_trade: Optional[TradeMetrics] = None

        # Для расчета награды
        self.last_portfolio_value = initial_balance
        self.peak_portfolio_value = initial_balance
        self.total_turnover = 0.0

        # Random state
        if random_seed is not None:
            self.seed(random_seed)

    def _setup_spaces(self):
        """Настроить observation и action spaces."""
        # Action space
        if self.action_type == ActionSpace.DISCRETE:
            # 0: Hold, 1: Buy, 2: Sell
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous: [-1, 1] где -1 = full sell, 0 = hold, 1 = full buy
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )

        # Observation space
        # Размер: окно данных + текущие позиции + метаданные
        feature_count = len(self.data_loader.get_feature_names())

        # Observation components:
        # 1. Price window (observation_window × features)
        # 2. Current positions (3: balance, crypto_held, total_value)
        # 3. Position state (2: position_ratio, unrealized_pnl_pct)
        # 4. Episode info (2: step_ratio, is_in_position)

        obs_size = (
            self.observation_window * feature_count +  # Historical data
            3 +  # Balance, crypto, portfolio value
            2 +  # Position ratio, unrealized PnL %
            2    # Step ratio, in position flag
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сбросить окружение.

        Args:
            seed: Random seed
            options: Дополнительные опции

        Returns:
            Tuple[observation, info]
        """
        super().reset(seed=seed)

        # Сброс состояния
        self.current_step = self.observation_window  # Начинаем после окна
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.equity_curve = [self.initial_balance]
        self.balance_history = [self.initial_balance]
        self.trades_history = []
        self.current_trade = None

        self.last_portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.total_turnover = 0.0

        # Сброс симулятора
        self.market_simulator.reset_history()

        # Получаем первое наблюдение
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: int | np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Выполнить шаг в окружении.

        Args:
            action: Действие агента

        Returns:
            Tuple[observation, reward, terminated, truncated, info]
        """
        if self.current_step >= len(self.data_loader):
            # Конец данных
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Текущая цена
        current_price = self.data_loader.get_price_at(self.current_step, 'close')

        # Выполняем действие
        trade_info = self._execute_action(action, current_price)

        # Обновляем портфель
        portfolio_value = self.balance + self.crypto_held * current_price
        self.equity_curve.append(portfolio_value)
        self.balance_history.append(self.balance)

        # Обновляем peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value

        # Рассчитываем награду
        reward = self._calculate_reward(trade_info, portfolio_value)

        # Обновляем последнее значение
        self.last_portfolio_value = portfolio_value

        # Переход к следующему шагу
        self.current_step += 1

        # Проверка условий завершения
        terminated = False
        truncated = False

        # Банкротство
        if self.stop_on_bankruptcy and portfolio_value <= self.bankruptcy_threshold:
            terminated = True

        # Максимальная просадка
        if self.max_drawdown_threshold is not None:
            current_dd = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value
            if current_dd <= self.max_drawdown_threshold:
                terminated = True

        # Максимальное количество шагов
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = True

        # Конец данных
        if self.current_step >= len(self.data_loader):
            truncated = True

        # Получаем наблюдение и информацию
        observation = self._get_observation()
        info = self._get_info()
        info['trade_executed'] = trade_info['executed']

        return observation, reward, terminated, truncated, info

    def _execute_action(self, action: int | np.ndarray, current_price: float) -> Dict[str, Any]:
        """
        Выполнить торговое действие.

        Args:
            action: Действие агента
            current_price: Текущая цена

        Returns:
            Словарь с информацией о сделке
        """
        trade_info = {
            'executed': False,
            'side': None,
            'quantity': 0.0,
            'price': current_price,
            'commission': 0.0,
            'slippage': 0.0
        }

        # Получаем состояние рынка
        volume = self.data_loader.data.iloc[self.current_step]['volume'] if 'volume' in self.data_loader.data.columns else 1000000.0
        market_state = self.market_simulator.get_market_state(
            mid_price=current_price,
            volume=volume
        )

        # Интерпретируем действие
        if self.action_type == ActionSpace.DISCRETE:
            if action == 0:  # Hold
                return trade_info
            elif action == 1:  # Buy
                # Покупаем на весь доступный баланс
                if self.balance > market_state.ask_price:
                    quantity = (self.balance * self.max_position_size) / market_state.ask_price
                    result = self.market_simulator.execute_order(
                        side=OrderSide.BUY,
                        quantity=quantity,
                        market_state=market_state
                    )

                    if result.executed:
                        cost = result.total_cost
                        self.balance -= cost
                        self.crypto_held += result.executed_quantity
                        self.total_trades += 1
                        self.total_turnover += cost

                        trade_info.update({
                            'executed': True,
                            'side': 'buy',
                            'quantity': result.executed_quantity,
                            'price': result.executed_price,
                            'commission': result.commission,
                            'slippage': result.slippage
                        })

                        # Начинаем новую сделку
                        if self.current_trade is None:
                            self.current_trade = TradeMetrics(
                                entry_time=self.current_step,
                                entry_price=result.executed_price,
                                quantity=result.executed_quantity,
                                side='long',
                                commission=result.commission
                            )

            elif action == 2:  # Sell
                # Продаем всю криптовалюту
                if self.crypto_held > 0:
                    quantity = self.crypto_held
                    result = self.market_simulator.execute_order(
                        side=OrderSide.SELL,
                        quantity=quantity,
                        market_state=market_state
                    )

                    if result.executed:
                        proceeds = result.executed_quantity * result.executed_price - result.commission
                        self.balance += proceeds
                        self.crypto_held -= result.executed_quantity
                        self.total_trades += 1
                        self.total_turnover += proceeds

                        trade_info.update({
                            'executed': True,
                            'side': 'sell',
                            'quantity': result.executed_quantity,
                            'price': result.executed_price,
                            'commission': result.commission,
                            'slippage': result.slippage
                        })

                        # Закрываем сделку
                        if self.current_trade is not None:
                            self.current_trade.exit_time = self.current_step
                            self.current_trade.exit_price = result.executed_price
                            self.current_trade.pnl = (
                                (result.executed_price - self.current_trade.entry_price) *
                                result.executed_quantity -
                                self.current_trade.commission -
                                result.commission
                            )
                            self.current_trade.pnl_pct = (
                                self.current_trade.pnl /
                                (self.current_trade.entry_price * self.current_trade.quantity) * 100
                            )
                            self.current_trade.holding_time = self.current_step - self.current_trade.entry_time
                            self.current_trade.is_winner = self.current_trade.pnl > 0
                            self.current_trade.commission += result.commission

                            self.trades_history.append(self.current_trade)
                            self.current_trade = None

        else:  # Continuous action space
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

            if action_value > 0.1:  # Buy signal
                # Размер позиции пропорционален action_value
                target_crypto_value = self.balance * self.max_position_size * action_value
                if target_crypto_value > self.crypto_held * current_price:
                    # Увеличиваем позицию
                    additional_value = target_crypto_value - self.crypto_held * current_price
                    quantity = additional_value / market_state.ask_price

                    if quantity > 0 and self.balance > additional_value:
                        result = self.market_simulator.execute_order(
                            side=OrderSide.BUY,
                            quantity=quantity,
                            market_state=market_state
                        )

                        if result.executed:
                            cost = result.total_cost
                            self.balance -= cost
                            self.crypto_held += result.executed_quantity
                            self.total_trades += 1
                            self.total_turnover += cost

                            trade_info.update({
                                'executed': True,
                                'side': 'buy',
                                'quantity': result.executed_quantity,
                                'price': result.executed_price,
                                'commission': result.commission,
                                'slippage': result.slippage
                            })

            elif action_value < -0.1:  # Sell signal
                # Продаем пропорционально abs(action_value)
                sell_ratio = min(abs(action_value), 1.0)
                quantity = self.crypto_held * sell_ratio

                if quantity > 0:
                    result = self.market_simulator.execute_order(
                        side=OrderSide.SELL,
                        quantity=quantity,
                        market_state=market_state
                    )

                    if result.executed:
                        proceeds = result.executed_quantity * result.executed_price - result.commission
                        self.balance += proceeds
                        self.crypto_held -= result.executed_quantity
                        self.total_trades += 1
                        self.total_turnover += proceeds

                        trade_info.update({
                            'executed': True,
                            'side': 'sell',
                            'quantity': result.executed_quantity,
                            'price': result.executed_price,
                            'commission': result.commission,
                            'slippage': result.slippage
                        })

        return trade_info

    def _calculate_reward(self, trade_info: Dict[str, Any], portfolio_value: float) -> float:
        """
        Рассчитать награду.

        Args:
            trade_info: Информация о сделке
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Награда
        """
        if self.reward_type == RewardType.PNL:
            # Простая разница в портфеле
            reward = portfolio_value - self.last_portfolio_value

        elif self.reward_type == RewardType.LOG_RETURN:
            # Log return
            if self.last_portfolio_value > 0:
                reward = np.log(portfolio_value / self.last_portfolio_value)
            else:
                reward = 0.0

        elif self.reward_type == RewardType.SHARPE:
            # Sharpe-подобная метрика (упрощенная)
            if len(self.equity_curve) > 2:
                returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
                mean_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 1.0
                reward = mean_return / (std_return + 1e-6)
            else:
                reward = 0.0

        elif self.reward_type == RewardType.SORTINO:
            # Sortino-подобная метрика
            if len(self.equity_curve) > 2:
                returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
                mean_return = np.mean(returns)
                downside_returns = returns[returns < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1.0
                reward = mean_return / (downside_std + 1e-6)
            else:
                reward = 0.0

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            # Risk-adjusted: PnL - λ·turnover - μ·drawdown
            pnl = portfolio_value - self.last_portfolio_value

            # Turnover penalty
            turnover_cost = 0.0
            if trade_info['executed']:
                turnover_cost = trade_info['quantity'] * trade_info['price'] * self.turnover_penalty

            # Drawdown penalty
            current_dd = (portfolio_value - self.peak_portfolio_value) / self.peak_portfolio_value
            drawdown_cost = abs(current_dd) * self.drawdown_penalty * self.initial_balance if current_dd < 0 else 0.0

            reward = pnl - turnover_cost - drawdown_cost

        else:
            reward = 0.0

        # Масштабирование
        reward *= self.reward_scaling

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """
        Получить текущее наблюдение.

        Returns:
            Numpy array с наблюдением
        """
        # 1. Historical price window
        window_data = self.data_loader.get_window(
            start_idx=self.current_step,
            window_size=self.observation_window
        )
        window_flat = window_data.flatten()

        # 2. Current portfolio state
        current_price = self.data_loader.get_price_at(
            min(self.current_step, len(self.data_loader) - 1),
            'close'
        )
        portfolio_value = self.balance + self.crypto_held * current_price

        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.crypto_held * current_price / self.initial_balance,  # Normalized crypto value
            portfolio_value / self.initial_balance  # Normalized total value
        ], dtype=np.float32)

        # 3. Position state
        position_ratio = (self.crypto_held * current_price) / portfolio_value if portfolio_value > 0 else 0.0

        if self.current_trade is not None:
            unrealized_pnl_pct = (
                (current_price - self.current_trade.entry_price) /
                self.current_trade.entry_price * 100
            )
        else:
            unrealized_pnl_pct = 0.0

        position_state = np.array([
            position_ratio,
            unrealized_pnl_pct / 100.0  # Нормализуем
        ], dtype=np.float32)

        # 4. Episode info
        step_ratio = self.current_step / len(self.data_loader)
        in_position = 1.0 if self.crypto_held > 0 else 0.0

        episode_info = np.array([
            step_ratio,
            in_position
        ], dtype=np.float32)

        # Объединяем все
        observation = np.concatenate([
            window_flat,
            portfolio_state,
            position_state,
            episode_info
        ]).astype(np.float32)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """Получить дополнительную информацию."""
        current_price = self.data_loader.get_price_at(
            min(self.current_step, len(self.data_loader) - 1),
            'close'
        )
        portfolio_value = self.balance + self.crypto_held * current_price

        return {
            'step': self.current_step,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value,
            'total_trades': self.total_trades,
            'current_price': current_price
        }

    def get_metrics(self) -> PerformanceMetrics:
        """
        Получить метрики производительности.

        Returns:
            PerformanceMetrics
        """
        return self.metrics_calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            trades=self.trades_history,
            initial_balance=self.initial_balance
        )

    def render(self):
        """Рендеринг окружения (базовая версия)."""
        if self.render_mode == "human":
            current_price = self.data_loader.get_price_at(
                min(self.current_step, len(self.data_loader) - 1),
                'close'
            )
            portfolio_value = self.balance + self.crypto_held * current_price

            print(f"\n=== Step {self.current_step} ===")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Crypto: {self.crypto_held:.6f}")
            print(f"Portfolio: ${portfolio_value:.2f}")
            print(f"Return: {((portfolio_value / self.initial_balance - 1) * 100):.2f}%")

    def seed(self, seed: int):
        """Установить random seed."""
        self.np_random = np.random.RandomState(seed)
        self.market_simulator.set_seed(seed)

    def close(self):
        """Закрыть окружение."""
        pass


if __name__ == "__main__":
    # Тестирование окружения
    print("=== Crypto Trading Environment Test ===\n")

    # Создаем окружение
    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        action_type=ActionSpace.DISCRETE,
        reward_type=RewardType.RISK_ADJUSTED,
        render_mode="human"
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Data loaded: {len(env.data_loader)} candles\n")

    # Тестовый эпизод
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}\n")

    # Несколько случайных шагов
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio: ${info['portfolio_value']:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            break

    # Финальные метрики
    metrics = env.get_metrics()
    print(f"\n=== Final Metrics ===")
    print(f"Total Return: {metrics.total_return_pct:.2f}%")
    print(f"Total Trades: {metrics.total_trades}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    env.close()
