"""
Торговая среда симуляции для DRL агентов.
Совместима с OpenAI Gym и Stable Baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """Типы действий в торговой среде."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingConfig:
    """Конфигурация торговой среды."""
    initial_balance: float = 10000.0  # Начальный баланс в USDT
    transaction_fee: float = 0.001  # Комиссия за транзакцию (0.1%)
    slippage: float = 0.0005  # Проскальзывание (0.05%)
    max_position_size: float = 1.0  # Максимальный размер позиции (100% баланса)
    min_trade_amount: float = 10.0  # Минимальная сумма сделки
    lookback_window: int = 50  # Количество исторических свечей для состояния
    

class TradingEnvironment(gym.Env):
    """
    Торговая среда для обучения DRL агентов.
    
    Пространство состояний: исторические данные + состояние портфеля
    Пространство действий: Hold/Buy/Sell + размер позиции
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        config: TradingConfig = None,
        reward_function: str = 'profit_based'
    ):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.config = config or TradingConfig()
        self.reward_function_name = reward_function
        
        self.logger = self._setup_logger()
        
        # Проверка данных
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Данные должны содержать колонки: {required_columns}")
        
        # Состояние среды
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        
        # История для аналитики
        self.portfolio_history = []
        self.trade_history = []
        self.action_history = []
        
        # Определение пространств
        self._define_spaces()
        
        self.logger.info(f"Торговая среда инициализирована с {len(self.data)} записями данных")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger('TradingEnvironment')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _define_spaces(self):
        """Определение пространств состояний и действий."""
        # Количество признаков в данных
        n_features = len(self.data.columns)
        
        # Пространство состояний: окно исторических данных + состояние портфеля
        # Форма: (lookback_window, n_features + portfolio_features)
        portfolio_features = 4  # balance, crypto_held, portfolio_value, portfolio_return
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.lookback_window, n_features + portfolio_features),
            dtype=np.float32
        )
        
        # Пространство действий: [action_type, position_size]
        # action_type: 0=Hold, 1=Buy, 2=Sell
        # position_size: доля от доступного баланса/позиции (0.0 - 1.0)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([2.0, 1.0]),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс среды к начальному состоянию."""
        super().reset(seed=seed)
        
        # Сброс состояния
        self.current_step = self.config.lookback_window
        self.balance = self.config.initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        
        # Очистка истории
        self.portfolio_history = []
        self.trade_history = []
        self.action_history = []
        
        # Начальное состояние портфеля
        initial_portfolio_value = self.balance
        self.portfolio_history.append(initial_portfolio_value)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Выполнение действия в среде."""
        # Извлечение действия
        action_type = int(np.clip(action[0], 0, 2))
        position_size = np.clip(action[1], 0.0, 1.0)
        
        # Получение текущей цены
        current_price = self.data.iloc[self.current_step]['close']
        
        # Выполнение торгового действия
        reward = self._execute_trade(action_type, position_size, current_price)
        
        # Обновление шага
        self.current_step += 1
        
        # Обновление истории портфеля
        portfolio_value = self._calculate_portfolio_value()
        self.portfolio_history.append(portfolio_value)
        
        # Сохранение действия
        self.action_history.append({
            'step': self.current_step,
            'action_type': action_type,
            'position_size': position_size,
            'price': current_price,
            'portfolio_value': portfolio_value
        })
        
        # Проверка завершения эпизода
        terminated = self.current_step >= len(self.data) - 1
        truncated = portfolio_value <= self.config.initial_balance * 0.1  # Остановка при больших потерях
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, action_type: int, position_size: float, current_price: float) -> float:
        """Выполнение торгового действия."""
        reward = 0.0
        
        if action_type == ActionType.BUY.value:
            # Покупка криптовалюты
            max_buy_amount = self.balance * position_size
            
            if max_buy_amount >= self.config.min_trade_amount:
                # Учет комиссии и проскальзывания
                effective_price = current_price * (1 + self.config.slippage)
                transaction_cost = max_buy_amount * self.config.transaction_fee
                net_amount = max_buy_amount - transaction_cost
                
                crypto_amount = net_amount / effective_price
                
                self.balance -= max_buy_amount
                self.crypto_held += crypto_amount
                self.total_trades += 1
                
                # Запись сделки
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'BUY',
                    'amount': crypto_amount,
                    'price': effective_price,
                    'cost': max_buy_amount,
                    'fee': transaction_cost
                })
                
                reward = -transaction_cost / self.config.initial_balance  # Небольшой штраф за комиссию
        
        elif action_type == ActionType.SELL.value:
            # Продажа криптовалюты
            crypto_to_sell = self.crypto_held * position_size
            
            if crypto_to_sell > 0:
                # Учет комиссии и проскальзывания
                effective_price = current_price * (1 - self.config.slippage)
                gross_amount = crypto_to_sell * effective_price
                transaction_cost = gross_amount * self.config.transaction_fee
                net_amount = gross_amount - transaction_cost
                
                self.balance += net_amount
                self.crypto_held -= crypto_to_sell
                self.total_trades += 1
                
                # Запись сделки
                self.trade_history.append({
                    'step': self.current_step,
                    'type': 'SELL',
                    'amount': crypto_to_sell,
                    'price': effective_price,
                    'revenue': net_amount,
                    'fee': transaction_cost
                })
                
                # Подсчет успешных сделок (упрощенно)
                if len(self.trade_history) >= 2:
                    last_buy = None
                    for trade in reversed(self.trade_history[:-1]):
                        if trade['type'] == 'BUY':
                            last_buy = trade
                            break
                    
                    if last_buy and effective_price > last_buy['price']:
                        self.successful_trades += 1
                
                reward = -transaction_cost / self.config.initial_balance  # Небольшой штраф за комиссию
        
        # Добавляем основную компоненту вознаграждения
        reward += self._calculate_reward()
        
        return reward
    
    def _calculate_reward(self) -> float:
        """Расчет вознаграждения на основе выбранной функции."""
        if self.reward_function_name == 'profit_based':
            return self._profit_based_reward()
        elif self.reward_function_name == 'sharpe_based':
            return self._sharpe_based_reward()
        elif self.reward_function_name == 'drawdown_penalty':
            return self._drawdown_penalty_reward()
        else:
            return self._profit_based_reward()
    
    def _profit_based_reward(self) -> float:
        """Вознаграждение на основе изменения стоимости портфеля."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        previous_value = self.portfolio_history[-2]
        current_value = self.portfolio_history[-1]
        
        # Относительное изменение
        if previous_value > 0:
            return (current_value - previous_value) / previous_value
        else:
            return 0.0
    
    def _sharpe_based_reward(self) -> float:
        """Вознаграждение на основе коэффициента Шарпа."""
        if len(self.portfolio_history) < 30:  # Минимум данных для расчета
            return self._profit_based_reward()
        
        # Расчет доходностей
        returns = []
        for i in range(1, len(self.portfolio_history)):
            if self.portfolio_history[i-1] > 0:
                ret = (self.portfolio_history[i] - self.portfolio_history[i-1]) / self.portfolio_history[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe_ratio = mean_return / std_return
            return sharpe_ratio * 0.01  # Масштабирование
        else:
            return mean_return
    
    def _drawdown_penalty_reward(self) -> float:
        """Вознаграждение с штрафом за просадку."""
        base_reward = self._profit_based_reward()
        
        if len(self.portfolio_history) < 2:
            return base_reward
        
        # Расчет текущей просадки
        peak_value = max(self.portfolio_history)
        current_value = self.portfolio_history[-1]
        
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            penalty = -drawdown * 2  # Штраф за просадку
            return base_reward + penalty
        
        return base_reward
    
    def _calculate_portfolio_value(self) -> float:
        """Расчет текущей стоимости портфеля."""
        current_price = self.data.iloc[self.current_step]['close']
        return self.balance + self.crypto_held * current_price
    
    def _get_observation(self) -> np.ndarray:
        """Получение текущего наблюдения."""
        # Исторические данные
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        
        historical_data = self.data.iloc[start_idx:end_idx].values
        
        # Если данных недостаточно, дополняем нулями
        if historical_data.shape[0] < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - historical_data.shape[0], historical_data.shape[1]))
            historical_data = np.vstack([padding, historical_data])
        
        # Состояние портфеля
        portfolio_value = self._calculate_portfolio_value()
        portfolio_return = (portfolio_value - self.config.initial_balance) / self.config.initial_balance
        
        portfolio_state = np.array([
            self.balance / self.config.initial_balance,  # Нормализованный баланс
            self.crypto_held * self.data.iloc[self.current_step]['close'] / self.config.initial_balance,  # Нормализованная позиция
            portfolio_value / self.config.initial_balance,  # Нормализованная стоимость портфеля
            portfolio_return  # Доходность портфеля
        ])
        
        # Дублирование состояния портфеля для каждого временного шага
        portfolio_features = np.tile(portfolio_state, (self.config.lookback_window, 1))
        
        # Объединение исторических данных и состояния портфеля
        observation = np.concatenate([historical_data, portfolio_features], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Получение дополнительной информации о состоянии среды."""
        portfolio_value = self._calculate_portfolio_value()
        
        return {
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value - self.config.initial_balance) / self.config.initial_balance,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate': self.successful_trades / max(self.total_trades, 1),
            'current_price': self.data.iloc[self.current_step]['close'],
            'step': self.current_step
        }
    
    def render(self, mode: str = 'human'):
        """Визуализация состояния среды."""
        if mode == 'human':
            info = self._get_info()
            print(f"Шаг: {info['step']}")
            print(f"Баланс: ${info['balance']:.2f}")
            print(f"Криптовалюта: {info['crypto_held']:.6f}")
            print(f"Стоимость портфеля: ${info['portfolio_value']:.2f}")
            print(f"Общая доходность: {info['total_return']:.2%}")
            print(f"Сделок: {info['total_trades']}")
            print(f"Винрейт: {info['win_rate']:.2%}")
            print(f"Текущая цена: ${info['current_price']:.2f}")
            print("-" * 50)


def create_trading_environment(
    data: pd.DataFrame,
    config: TradingConfig = None,
    reward_function: str = 'profit_based'
) -> TradingEnvironment:
    """Удобная функция для создания торговой среды."""
    return TradingEnvironment(data, config, reward_function)


def main():
    """Пример использования торговой среды."""
    # Создание тестовых данных
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Обеспечение логичности OHLC данных
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    # Создание среды
    config = TradingConfig(initial_balance=10000.0, lookback_window=20)
    env = create_trading_environment(test_data, config)
    
    print("Тестирование торговой среды...")
    
    # Тестовый эпизод
    observation, info = env.reset()
    print(f"Начальное наблюдение: {observation.shape}")
    print(f"Начальная информация: {info}")
    
    # Несколько случайных действий
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Шаг {i+1}:")
        print(f"  Действие: {action}")
        print(f"  Вознаграждение: {reward:.4f}")
        print(f"  Портфель: ${info['portfolio_value']:.2f}")
        print(f"  Доходность: {info['total_return']:.2%}")
        
        if terminated or truncated:
            break
    
    print("Тест завершен!")


if __name__ == "__main__":
    main()