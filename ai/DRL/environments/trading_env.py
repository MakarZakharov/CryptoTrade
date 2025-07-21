"""Основная торговая среда для DRL обучения."""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union
from datetime import datetime

from ..config import TradingConfig
from ..data import CSVDataLoader, DataPreprocessor, TechnicalIndicators, DataValidator
from ..utils import DRLLogger, TradingMetrics
from .portfolio_manager import PortfolioManager
from .reward_calculator import RewardCalculator
from .market_simulator import MarketSimulator


class TradingEnv(gym.Env):
    """
    Gymnasium-совместимая торговая среда для криптовалют.
    
    Эта среда симулирует торговлю криптовалютами с реалистичными
    рыночными условиями, включая комиссии, проскальзывание и управление рисками.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 4
    }
    
    def __init__(
        self,
        config: TradingConfig,
        data: Optional[pd.DataFrame] = None,
        render_mode: Optional[str] = None,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация торговой среды.
        
        Args:
            config: конфигурация торговых параметров
            data: предзагруженные данные (опционально)
            render_mode: режим рендеринга
            logger: логгер для записи операций
        """
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        self.logger = logger or DRLLogger(f"trading_env_{config.symbol}")
        
        # Инициализация компонентов
        self._init_data_components(data)
        self._init_trading_components()
        self._init_spaces()
        
        # Состояние среды
        self.current_step = 0
        self.episode_start_step = 0
        self.episode_count = 0
        self.is_initialized = False
        
        # Рендеринг
        self.render_history = []
        self.window = None
        self.clock = None
        
        self.logger.info(f"Торговая среда инициализирована для {config.symbol}")
    
    def _init_data_components(self, data: Optional[pd.DataFrame]):
        """Инициализация компонентов обработки данных."""
        # Загрузчик данных
        self.data_loader = CSVDataLoader(self.config.data_dir, self.logger)
        
        # Предобработчик данных
        self.preprocessor = DataPreprocessor(self.config, self.logger)
        
        # Технические индикаторы
        self.technical_indicators = TechnicalIndicators(self.logger)
        
        # Валидатор данных
        self.data_validator = DataValidator(self.logger)
        
        # Загрузка и подготовка данных
        if data is not None:
            self.raw_data = data.copy()
        else:
            self.raw_data = self.data_loader.load_data(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe
            )
        
        # Валидация данных
        validation_result = self.data_validator.validate_ohlcv(self.raw_data)
        if not validation_result['is_valid']:
            raise ValueError(f"Данные не прошли валидацию: {validation_result['errors']}")
        
        # Добавление технических индикаторов
        if self.config.include_technical_indicators:
            self.raw_data = self.technical_indicators.add_all_indicators(self.raw_data)
        
        # Предобработка данных
        self.processed_data = self.preprocessor.prepare_for_drl(self.raw_data)
        
        # Разделение на train/val/test
        self.train_data, self.val_data, self.test_data = self.preprocessor.split_data_for_training(
            self.processed_data
        )
        
        # Установка активного датасета (по умолчанию train)
        self.active_data = self.train_data
        
        self.logger.info(f"Данные подготовлены: train={len(self.train_data)}, val={len(self.val_data)}, test={len(self.test_data)}")
    
    def _init_trading_components(self):
        """Инициализация торговых компонентов."""
        # Менеджер портфеля
        self.portfolio_manager = PortfolioManager(self.config, self.logger)
        
        # Калькулятор наград
        self.reward_calculator = RewardCalculator(self.config, self.logger)
        
        # Симулятор рынка
        self.market_simulator = MarketSimulator(self.config, self.logger)
        
        # Метрики торговли
        self.trading_metrics = TradingMetrics()
    
    def _init_spaces(self):
        """Инициализация пространств действий и наблюдений."""
        # Пространство действий
        action_info = self.config.get_action_space_info()
        
        if action_info["type"] == "continuous":
            self.action_space = gym.spaces.Box(
                low=action_info["low"],
                high=action_info["high"],
                shape=action_info["shape"],
                dtype=np.float32
            )
        else:  # discrete
            self.action_space = gym.spaces.Discrete(action_info["n"])
        
        # Пространство наблюдений
        # Рассчитываем размер на основе данных и конфигурации
        observation_size = self._calculate_observation_size()
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32
        )
        
        self.logger.debug(f"Пространство действий: {self.action_space}")
        self.logger.debug(f"Пространство наблюдений: {self.observation_space.shape}")
    
    def _calculate_observation_size(self) -> int:
        """Расчет размера пространства наблюдений."""
        # Получаем точное количество столбцов из обработанных данных
        market_features_per_step = len(self.processed_data.columns)
        
        # Фичи портфеля (фиксированное количество)
        portfolio_features = 5  # balance_usdt, balance_crypto, total_value, position_size, unrealized_pnl
        
        # Учитываем lookback window для исторических данных
        historical_size = market_features_per_step * self.config.lookback_window
        
        total_size = historical_size + portfolio_features
        
        self.logger.debug(f"Размер наблюдения: рыночные_фичи_за_шаг={market_features_per_step}, "
                         f"портфель={portfolio_features}, окно={self.config.lookback_window}, "
                         f"итого={total_size}")
        
        return total_size
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Сброс среды в начальное состояние.
        
        Args:
            seed: сид для воспроизводимости
            options: дополнительные опции
            
        Returns:
            Tuple[observation, info] - начальное наблюдение и информация
        """
        # Обязательный вызов для правильной инициализации RNG
        super().reset(seed=seed)
        
        # Обработка опций
        if options is not None:
            if "data_split" in options:
                self._set_data_split(options["data_split"])
            if "start_step" in options:
                self.episode_start_step = options["start_step"]
        
        # Сброс состояния
        self._reset_episode_state()
        
        # Получение начального наблюдения
        observation = self._get_observation()
        info = self._get_info()
        
        self.is_initialized = True
        self.episode_count += 1
        
        self.logger.debug(f"Эпизод {self.episode_count} начат с шага {self.current_step}")
        
        return observation, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Выполнение одного шага в среде.
        
        Args:
            action: действие агента
            
        Returns:
            Tuple[observation, reward, terminated, truncated, info]
        """
        if not self.is_initialized:
            raise RuntimeError("Среда не инициализирована. Вызовите reset() перед step().")
        
        # Проверка валидности действия
        if not self.action_space.contains(action):
            raise ValueError(f"Недопустимое действие: {action}")
        
        # Сохранение предыдущего состояния для расчета награды
        prev_portfolio_value = self.portfolio_manager.get_total_value()
        
        # Выполнение действия
        trade_info = self._execute_action(action)
        
        # Обновление состояния среды
        self._update_environment_state()
        
        # Расчет награды
        reward = self._calculate_reward(prev_portfolio_value, trade_info)
        
        # Проверка условий завершения
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # Получение нового наблюдения
        observation = self._get_observation()
        info = self._get_info(trade_info, reward)
        
        # Добавляем episode info для stable_baselines3 только при завершении эпизода
        if terminated or truncated:
            # Stable-Baselines3 expects EXACTLY this format for episode info
            # Must be a dict with 'r' (float), 'l' (int), 't' (float) - no other keys
            episode_reward = self.portfolio_manager.get_total_return()
            episode_length = self.current_step - self.episode_start_step
            
            # Critical: SB3 Monitor wrapper expects exactly these keys and types
            info['episode'] = {
                'r': float(episode_reward),  # episodic reward - MUST be float
                'l': int(episode_length),    # episode length - MUST be int  
                't': float(self.current_step)  # total timesteps - MUST be float
            }
            
            # This flag tells SB3 Monitor that episode has ended
            info['_episode'] = True
        
        # Переход к следующему шагу
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """Выполнение торгового действия."""
        current_price = self._get_current_price()
        
        # Симуляция рыночных условий
        execution_price = self.market_simulator.simulate_execution(
            action, current_price, self._get_current_volume()
        )
        
        # Выполнение торговли через менеджер портфеля
        trade_info = self.portfolio_manager.execute_trade(
            action=action,
            price=execution_price,
            timestamp=self._get_current_timestamp()
        )
        
        return trade_info
    
    def _update_environment_state(self):
        """Обновление состояния среды."""
        current_price = self._get_current_price()
        self.portfolio_manager.update_portfolio_value(current_price)
        
        # Обновление истории для рендеринга
        if self.render_mode is not None:
            self.render_history.append({
                'step': self.current_step,
                'price': current_price,
                'portfolio_value': self.portfolio_manager.get_total_value(),
                'position': self.portfolio_manager.position_size,
                'timestamp': self._get_current_timestamp()
            })
    
    def _calculate_reward(self, prev_portfolio_value: float, trade_info: Dict[str, Any]) -> float:
        """Расчет награды за действие."""
        return self.reward_calculator.calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            current_portfolio_value=self.portfolio_manager.get_total_value(),
            trade_info=trade_info,
            current_step=self.current_step,
            market_data=self._get_current_market_data()
        )
    
    def _check_terminated(self) -> bool:
        """Проверка условий успешного завершения эпизода."""
        # Завершение при достижении целевой доходности
        total_return = self.portfolio_manager.get_total_return()
        if total_return >= self.config.target_monthly_return:
            self.logger.info(f"Цель достигнута! Доходность: {total_return:.4f}")
            return True
        
        # Завершение при критической просадке
        if self.portfolio_manager.get_current_drawdown() >= self.config.early_stopping_loss:
            self.logger.warning(f"Критическая просадка! Завершение эпизода.")
            return True
        
        return False
    
    def _check_truncated(self) -> bool:
        """Проверка условий принудительного завершения эпизода."""
        # Превышение максимального количества шагов
        if self.current_step - self.episode_start_step >= self.config.max_episode_steps:
            return True
        
        # Достижение конца данных
        if self.current_step >= len(self.active_data) - 1:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Получение текущего наблюдения."""
        # Проверка границ данных
        if self.current_step < self.config.lookback_window:
            # Дополняем начальными значениями
            historical_data = self.active_data.iloc[:self.current_step + 1]
            # Дублируем первую строку для заполнения недостающих данных
            padding_needed = self.config.lookback_window - len(historical_data)
            if padding_needed > 0:
                padding = pd.concat([historical_data.iloc[:1]] * padding_needed)
                historical_data = pd.concat([padding, historical_data])
        else:
            historical_data = self.active_data.iloc[
                self.current_step - self.config.lookback_window + 1:self.current_step + 1
            ]
        
        # Преобразование в numpy массив
        market_features = historical_data.values.flatten().astype(np.float32)
        
        # Добавление состояния портфеля
        portfolio_features = np.array([
            self.portfolio_manager.balance_usdt / self.config.initial_balance,  # Нормализованный баланс USDT
            self.portfolio_manager.balance_crypto,  # Баланс криптовалюты
            self.portfolio_manager.get_total_value() / self.config.initial_balance,  # Нормализованная общая стоимость
            self.portfolio_manager.position_size,  # Размер позиции
            self.portfolio_manager.get_unrealized_pnl() / self.config.initial_balance  # Нормализованная нереализованная прибыль/убыток
        ], dtype=np.float32)
        
        # Объединение всех фичей
        observation = np.concatenate([market_features, portfolio_features])
        
        # Проверка размера наблюдения
        expected_size = self.observation_space.shape[0]
        if len(observation) != expected_size:
            self.logger.warning(f"Несоответствие размера наблюдения: получено {len(observation)}, ожидалось {expected_size}")
            # Обрезаем или дополняем до нужного размера
            if len(observation) > expected_size:
                observation = observation[:expected_size]
            else:
                padding = np.zeros(expected_size - len(observation), dtype=np.float32)
                observation = np.concatenate([observation, padding])
            self.logger.debug(f"Размер наблюдения исправлен: {len(observation)}")
        
        return observation
    
    def _get_info(self, trade_info: Optional[Dict[str, Any]] = None, reward: Optional[float] = None) -> Dict[str, Any]:
        """Получение дополнительной информации о состоянии среды."""
        try:
            info = {
                'step': int(self.current_step),
                'episode': int(self.episode_count),
                'timestamp': str(self._get_current_timestamp()),
                'price': float(self._get_current_price()),
                'portfolio': {
                    'total_value': float(self.portfolio_manager.get_total_value()),
                    'balance_usdt': float(self.portfolio_manager.balance_usdt),
                    'balance_crypto': float(self.portfolio_manager.balance_crypto),
                    'position_size': float(self.portfolio_manager.position_size),
                    'total_return': float(self.portfolio_manager.get_total_return()),
                    'realized_pnl': float(self.portfolio_manager.realized_pnl),
                    'unrealized_pnl': float(self.portfolio_manager.get_unrealized_pnl()),
                    'drawdown': float(self.portfolio_manager.get_current_drawdown()),
                    'trades_count': int(len(self.portfolio_manager.trade_history))
                }
            }
            
            if trade_info is not None and isinstance(trade_info, dict):
                info['trade'] = trade_info
            
            if reward is not None:
                info['reward'] = float(reward)
            
            # Добавляем метрики производительности
            if len(self.portfolio_manager.trade_history) > 0:
                returns = [trade.pnl for trade in self.portfolio_manager.trade_history if trade.pnl != 0]
                if returns:
                    info['metrics'] = {
                        'sharpe_ratio': float(self.trading_metrics.sharpe_ratio(returns)),
                        'win_rate': float(self.trading_metrics.win_rate(returns)),
                        'profit_factor': float(self.trading_metrics.profit_factor(returns))
                    }
                else:
                    info['metrics'] = {'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
            else:
                info['metrics'] = {'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0}
            
            return info
        except Exception as e:
            self.logger.error(f"Ошибка в _get_info: {e}")
            # Возвращаем минимальную безопасную версию info
            return {
                'step': int(self.current_step),
                'episode': int(self.episode_count),
                'error': str(e)
            }
    
    def _reset_episode_state(self):
        """Сброс состояния эпизода."""
        # Случайное начальное положение (если не указано явно)
        if hasattr(self, 'episode_start_step') and self.episode_start_step == 0:
            max_start = len(self.active_data) - self.config.max_episode_steps - self.config.lookback_window
            if max_start > 0:
                self.episode_start_step = self.np_random.integers(0, max_start)
        
        self.current_step = self.episode_start_step
        
        # Сброс компонентов
        self.portfolio_manager.reset(self.config.initial_balance)
        self.reward_calculator.reset()
        self.render_history.clear()
    
    def _set_data_split(self, split: str):
        """Установка активного раздела данных."""
        if split == "train":
            self.active_data = self.train_data
        elif split == "val":
            self.active_data = self.val_data
        elif split == "test":
            self.active_data = self.test_data
        else:
            raise ValueError(f"Неизвестный раздел данных: {split}")
        
        self.logger.debug(f"Переключение на {split} данные ({len(self.active_data)} записей)")
    
    def _get_current_price(self) -> float:
        """Получение текущей цены."""
        if self.current_step >= len(self.active_data):
            return self.active_data.iloc[-1]['close']
        return float(self.active_data.iloc[self.current_step]['close'])
    
    def _get_current_volume(self) -> float:
        """Получение текущего объема."""
        if self.current_step >= len(self.active_data):
            return self.active_data.iloc[-1]['volume']
        return float(self.active_data.iloc[self.current_step]['volume'])
    
    def _get_current_timestamp(self) -> str:
        """Получение текущего времени."""
        if self.current_step >= len(self.active_data):
            return str(self.active_data.index[-1])
        return str(self.active_data.index[self.current_step])
    
    def _get_current_market_data(self) -> Dict[str, float]:
        """Получение текущих рыночных данных."""
        if self.current_step >= len(self.active_data):
            row = self.active_data.iloc[-1]
        else:
            row = self.active_data.iloc[self.current_step]
        
        return {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        }
    
    def render(self) -> Optional[Union[str, np.ndarray]]:
        """Рендеринг текущего состояния среды."""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None
    
    def _render_human(self) -> None:
        """Рендеринг для человека."""
        if len(self.render_history) == 0:
            return
        
        current_data = self.render_history[-1]
        print(f"\n{'='*60}")
        print(f"Шаг: {current_data['step']} | Время: {current_data['timestamp']}")
        print(f"Цена: ${current_data['price']:.4f}")
        print(f"Стоимость портфеля: ${current_data['portfolio_value']:.2f}")
        print(f"Позиция: {current_data['position']:.4f}")
        print(f"Доходность: {self.portfolio_manager.get_total_return()*100:.2f}%")
        print(f"Просадка: {self.portfolio_manager.get_current_drawdown()*100:.2f}%")
        print(f"{'='*60}")
    
    def _render_ansi(self) -> str:
        """Рендеринг в ANSI формате."""
        if len(self.render_history) == 0:
            return "Нет данных для отображения"
        
        current_data = self.render_history[-1]
        return (f"Step: {current_data['step']} | "
                f"Price: ${current_data['price']:.4f} | "
                f"Portfolio: ${current_data['portfolio_value']:.2f} | "
                f"Return: {self.portfolio_manager.get_total_return()*100:.2f}%")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Рендеринг в RGB массив (для визуализации графиков)."""
        # Простая реализация - возвращаем черный квадрат
        # В реальной реализации здесь был бы график цены и портфеля
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Закрытие среды и освобождение ресурсов."""
        if self.window is not None:
            # Закрытие окна рендеринга (если используется)
            pass
        
        self.logger.info("Торговая среда закрыта")
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Получение сводки по завершенному эпизоду."""
        if not self.portfolio_manager.trade_history:
            return {"message": "Торгов не было"}
        
        returns = [trade.pnl for trade in self.portfolio_manager.trade_history if trade.pnl != 0]
        
        return {
            'total_steps': self.current_step - self.episode_start_step,
            'total_trades': len(self.portfolio_manager.trade_history),
            'final_portfolio_value': self.portfolio_manager.get_total_value(),
            'total_return': self.portfolio_manager.get_total_return(),
            'max_drawdown': self.portfolio_manager.get_max_drawdown(),
            'realized_pnl': self.portfolio_manager.realized_pnl,
            'metrics': {
                'sharpe_ratio': self.trading_metrics.sharpe_ratio(returns) if returns else 0,
                'win_rate': self.trading_metrics.win_rate(returns) if returns else 0,
                'profit_factor': self.trading_metrics.profit_factor(returns) if returns else 0
            } if returns else {}
        }
    
    def set_data_split(self, split: str):
        """Публичный метод для установки раздела данных."""
        self._set_data_split(split)
    
    def get_data_info(self) -> Dict[str, Any]:
        """Получение информации о данных."""
        return {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'total_samples': len(self.processed_data),
            'train_samples': len(self.train_data),
            'val_samples': len(self.val_data),  
            'test_samples': len(self.test_data),
            'features': list(self.processed_data.columns),
            'lookback_window': self.config.lookback_window,
            'action_space': str(self.action_space),
            'observation_space': str(self.observation_space)
        }