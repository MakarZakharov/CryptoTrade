"""Симулятор рыночных условий для торговой среды."""

import numpy as np
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import random

from ..config import TradingConfig
from ..utils import DRLLogger


class MarketCondition(Enum):
    """Рыночные условия."""
    BULL = "bull"          # Бычий тренд
    BEAR = "bear"          # Медвежий тренд
    SIDEWAYS = "sideways"  # Боковой тренд
    VOLATILE = "volatile"  # Высокая волатильность


class OrderType(Enum):
    """Типы ордеров."""
    MARKET = "market"      # Рыночный ордер
    LIMIT = "limit"        # Лимитный ордер
    STOP = "stop"          # Стоп-ордер


class MarketSimulator:
    """
    Симулятор рыночных условий для реалистичной торговой среды.
    
    Симулирует:
    - Проскальзывание (slippage)
    - Спреды bid/ask
    - Влияние объема на исполнение
    - Рыночные микроструктуры
    - Изменяющиеся рыночные условия
    """
    
    def __init__(self, config: TradingConfig, logger: Optional[DRLLogger] = None):
        """
        Инициализация симулятора рынка.
        
        Args:
            config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        self.config = config
        self.logger = logger or DRLLogger("market_simulator")
        
        # Параметры рынка
        self.commission_rate = config.commission_rate
        self.slippage_rate = config.slippage_rate
        self.spread_rate = config.spread_rate
        
        # Состояние рынка
        self.current_condition = MarketCondition.SIDEWAYS
        self.volatility_factor = 1.0
        self.liquidity_factor = 1.0
        
        # История для адаптации
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        # Параметры адаптации
        self.min_lookback = 20
        self.max_lookback = 100
        
        # Рыночные микроструктуры
        self.bid_ask_spread = 0.0
        self.market_depth = 1000000.0  # Глубина рынка в USDT
        self.last_trade_price = 0.0
        
        self.logger.info("Симулятор рынка инициализирован")
    
    def update_market_state(self, current_price: float, current_volume: float):
        """
        Обновление состояния рынка на основе текущих данных.
        
        Args:
            current_price: текущая цена
            current_volume: текущий объем
        """
        # Обновление истории
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # Ограничение размера истории
        if len(self.price_history) > self.max_lookback:
            self.price_history = self.price_history[-self.max_lookback:]
            self.volume_history = self.volume_history[-self.max_lookback:]
        
        # Обновление рыночных условий
        self._update_market_condition()
        self._update_volatility_factor()
        self._update_liquidity_factor()
        self._update_bid_ask_spread(current_price)
        
        self.last_trade_price = current_price
    
    def simulate_execution(
        self,
        action: Union[int, np.ndarray, float],
        market_price: float,
        market_volume: float,
        order_type: OrderType = OrderType.MARKET
    ) -> float:
        """
        Симуляция исполнения ордера с учетом рыночных условий.
        
        Args:
            action: действие агента
            market_price: рыночная цена
            market_volume: рыночный объем
            order_type: тип ордера
            
        Returns:
            Цена исполнения с учетом всех факторов
        """
        # Обновление состояния рынка
        self.update_market_state(market_price, market_volume)
        
        # Определение направления сделки
        trade_direction = self._get_trade_direction(action)
        
        if trade_direction == "hold":
            return market_price
        
        # Базовая цена исполнения
        execution_price = market_price
        
        # Применение спреда bid/ask
        execution_price = self._apply_bid_ask_spread(execution_price, trade_direction)
        
        # Применение проскальзывания
        execution_price = self._apply_slippage(execution_price, trade_direction, market_volume)
        
        # Применение влияния объема (market impact)
        execution_price = self._apply_market_impact(execution_price, trade_direction, action)
        
        # Симуляция частичного исполнения в экстремальных условиях
        execution_info = self._simulate_execution_quality(execution_price, market_price, trade_direction)
        
        # Сохранение информации об исполнении
        self._record_execution(action, market_price, execution_price, execution_info)
        
        self.logger.debug(f"Исполнение: направление={trade_direction}, рынок=${market_price:.4f}, "
                         f"исполнение=${execution_price:.4f}, условия={self.current_condition.value}")
        
        return execution_price
    
    def _get_trade_direction(self, action: Union[int, np.ndarray, float]) -> str:
        """Определение направления сделки из действия."""
        if self.config.action_type == "continuous":
            if isinstance(action, np.ndarray):
                action = float(action[0])
            
            if abs(action) < 0.1:
                return "hold"
            elif action > 0:
                return "buy"
            else:
                return "sell"
        else:  # discrete
            action_map = {0: "buy", 1: "sell", 2: "hold"}
            return action_map.get(int(action), "hold")
    
    def _apply_bid_ask_spread(self, price: float, direction: str) -> float:
        """Применение спреда bid/ask."""
        if direction == "hold":
            return price
        
        # Рассчитываем текущий спред
        spread = self.bid_ask_spread
        
        if direction == "buy":
            # Покупаем по ask (более высокой цене)
            return price * (1 + spread / 2)
        else:  # sell
            # Продаем по bid (более низкой цене)
            return price * (1 - spread / 2)
    
    def _apply_slippage(self, price: float, direction: str, volume: float) -> float:
        """Применение проскальзывания."""
        if direction == "hold":
            return price
        
        # Базовое проскальзывание
        base_slippage = self.slippage_rate
        
        # Увеличение проскальзывания в зависимости от условий рынка
        condition_multiplier = {
            MarketCondition.BULL: 0.8,      # Меньше проскальзывания в бычьем тренде
            MarketCondition.BEAR: 1.2,      # Больше проскальзывания в медвежьем тренде
            MarketCondition.SIDEWAYS: 1.0,  # Нормальное проскальзывание
            MarketCondition.VOLATILE: 1.5   # Значительное проскальзывание при волатильности
        }
        
        # Влияние волатильности
        volatility_multiplier = 0.5 + self.volatility_factor
        
        # Влияние ликвидности
        liquidity_multiplier = 2.0 - self.liquidity_factor
        
        # Общее проскальзывание
        total_slippage = (base_slippage * 
                         condition_multiplier[self.current_condition] * 
                         volatility_multiplier * 
                         liquidity_multiplier)
        
        # Случайный компонент
        random_factor = random.uniform(0.5, 1.5)
        total_slippage *= random_factor
        
        # Ограничение максимального проскальзывания
        total_slippage = min(total_slippage, 0.01)  # Максимум 1%
        
        if direction == "buy":
            return price * (1 + total_slippage)
        else:  # sell
            return price * (1 - total_slippage)
    
    def _apply_market_impact(self, price: float, direction: str, action: Union[int, np.ndarray, float]) -> float:
        """Применение влияния на рынок (market impact)."""
        if direction == "hold":
            return price
        
        # Размер ордера (в процентах от капитала)
        if self.config.action_type == "continuous":
            if isinstance(action, np.ndarray):
                action = float(action[0])
            order_size = abs(action)
        else:
            order_size = 0.5  # Фиксированный размер для дискретных действий
        
        # Базовое влияние на рынок
        base_impact = order_size * 0.001  # 0.1% за полный размер
        
        # Модификация в зависимости от ликвидности
        liquidity_impact = base_impact * (2.0 - self.liquidity_factor)
        
        # Модификация в зависимости от волатильности
        volatility_impact = liquidity_impact * (1.0 + self.volatility_factor * 0.5)
        
        # Ограничение максимального влияния
        final_impact = min(volatility_impact, 0.005)  # Максимум 0.5%
        
        if direction == "buy":
            return price * (1 + final_impact)
        else:  # sell
            return price * (1 - final_impact)
    
    def _simulate_execution_quality(self, execution_price: float, market_price: float, direction: str) -> Dict[str, Any]:
        """Симуляция качества исполнения."""
        # Процент исполнения (обычно 100%, но может быть меньше в экстремальных условиях)
        fill_percentage = 1.0
        
        # В условиях высокой волатильности или низкой ликвидности
        if (self.current_condition == MarketCondition.VOLATILE or 
            self.liquidity_factor < 0.3):
            fill_percentage = random.uniform(0.8, 1.0)
        
        # Время исполнения (в миллисекундах)
        execution_time = self._calculate_execution_time()
        
        # Отклонение от рыночной цены
        price_deviation = abs(execution_price - market_price) / market_price
        
        return {
            "fill_percentage": fill_percentage,
            "execution_time_ms": execution_time,
            "price_deviation": price_deviation,
            "market_condition": self.current_condition.value,
            "liquidity_factor": self.liquidity_factor,
            "volatility_factor": self.volatility_factor
        }
    
    def _calculate_execution_time(self) -> float:
        """Расчет времени исполнения в миллисекундах."""
        # Базовое время исполнения
        base_time = 50.0  # 50ms
        
        # Увеличение времени в зависимости от условий
        condition_multiplier = {
            MarketCondition.BULL: 0.8,
            MarketCondition.BEAR: 1.2,
            MarketCondition.SIDEWAYS: 1.0,
            MarketCondition.VOLATILE: 2.0
        }
        
        # Влияние ликвидности
        liquidity_multiplier = 3.0 - 2.0 * self.liquidity_factor
        
        total_time = (base_time * 
                     condition_multiplier[self.current_condition] * 
                     liquidity_multiplier)
        
        # Добавление случайности
        total_time *= random.uniform(0.5, 2.0)
        
        return total_time
    
    def _update_market_condition(self):
        """Обновление рыночных условий на основе истории цен."""
        if len(self.price_history) < self.min_lookback:
            return
        
        recent_prices = self.price_history[-self.min_lookback:]
        
        # Расчет трендов
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
        
        # Определение условий
        if volatility > 0.05:  # Высокая волатильность (>5%)
            self.current_condition = MarketCondition.VOLATILE
        elif price_change > 0.02:  # Рост >2%
            self.current_condition = MarketCondition.BULL
        elif price_change < -0.02:  # Падение >2%
            self.current_condition = MarketCondition.BEAR
        else:
            self.current_condition = MarketCondition.SIDEWAYS
    
    def _update_volatility_factor(self):
        """Обновление фактора волатильности."""
        if len(self.price_history) < 5:
            self.volatility_factor = 1.0
            return
        
        # Берем последние цены для расчета доходности
        recent_prices = self.price_history[-20:] if len(self.price_history) >= 20 else self.price_history
        
        if len(recent_prices) < 2:
            self.volatility_factor = 1.0
            return
            
        # Рассчитываем доходности
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        current_volatility = np.std(recent_returns)
        
        # Нормализация волатильности (среднее значение = 1.0)
        # Типичная дневная волатильность криптовалют ~ 3-5%
        normal_volatility = 0.04
        self.volatility_factor = min(current_volatility / normal_volatility, 3.0)
    
    def _update_liquidity_factor(self):
        """Обновление фактора ликвидности."""
        if len(self.volume_history) < 5:
            self.liquidity_factor = 1.0
            return
        
        recent_volumes = self.volume_history[-10:]
        current_volume = recent_volumes[-1]
        avg_volume = np.mean(recent_volumes[:-1])
        
        if avg_volume > 0:
            # Высокий объем = высокая ликвидность
            self.liquidity_factor = min(current_volume / avg_volume, 2.0)
        else:
            self.liquidity_factor = 1.0
    
    def _update_bid_ask_spread(self, current_price: float):
        """Обновление спреда bid/ask."""
        # Базовый спред
        base_spread = self.spread_rate
        
        # Влияние волатильности
        volatility_spread = base_spread * self.volatility_factor
        
        # Влияние ликвидности (обратная зависимость)
        liquidity_spread = volatility_spread / self.liquidity_factor
        
        # Влияние рыночных условий
        condition_multiplier = {
            MarketCondition.BULL: 0.9,
            MarketCondition.BEAR: 1.1,
            MarketCondition.SIDEWAYS: 1.0,
            MarketCondition.VOLATILE: 1.5
        }
        
        self.bid_ask_spread = (liquidity_spread * 
                              condition_multiplier[self.current_condition])
        
        # Ограничение максимального спреда
        self.bid_ask_spread = min(self.bid_ask_spread, 0.005)  # Максимум 0.5%
    
    def _record_execution(self, action: Any, market_price: float, execution_price: float, execution_info: Dict[str, Any]):
        """Запись информации об исполнении."""
        record = {
            "action": str(action),
            "market_price": market_price,
            "execution_price": execution_price,
            "slippage": abs(execution_price - market_price) / market_price,
            "market_condition": self.current_condition.value,
            **execution_info
        }
        
        self.execution_history.append(record)
        
        # Ограничение размера истории
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
    
    def get_market_state(self) -> Dict[str, Any]:
        """Получение текущего состояния рынка."""
        return {
            "condition": self.current_condition.value,
            "volatility_factor": self.volatility_factor,
            "liquidity_factor": self.liquidity_factor,
            "bid_ask_spread": self.bid_ask_spread,
            "last_price": self.last_trade_price
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Получение статистики исполнения."""
        if not self.execution_history:
            return {"total_executions": 0}
        
        slippages = [record["slippage"] for record in self.execution_history]
        fill_percentages = [record.get("fill_percentage", 1.0) for record in self.execution_history]
        execution_times = [record.get("execution_time_ms", 0) for record in self.execution_history]
        
        stats = {
            "total_executions": len(self.execution_history),
            "avg_slippage": np.mean(slippages),
            "max_slippage": np.max(slippages),
            "avg_fill_percentage": np.mean(fill_percentages),
            "avg_execution_time_ms": np.mean(execution_times),
            "market_conditions": {}
        }
        
        # Статистика по рыночным условиям
        for condition in MarketCondition:
            condition_records = [r for r in self.execution_history 
                               if r["market_condition"] == condition.value]
            if condition_records:
                condition_slippages = [r["slippage"] for r in condition_records]
                stats["market_conditions"][condition.value] = {
                    "count": len(condition_records),
                    "avg_slippage": np.mean(condition_slippages)
                }
        
        return stats
    
    def simulate_market_hours(self) -> Dict[str, Any]:
        """Симуляция влияния рабочих часов рынка."""
        # Криптовалютные рынки работают 24/7, но есть периоды активности
        import datetime
        
        current_hour = datetime.datetime.now().hour
        
        # Периоды активности (UTC)
        # Азиатская сессия: 00-08
        # Европейская сессия: 08-16  
        # Американская сессия: 16-24
        
        if 8 <= current_hour < 16:  # Европейская сессия
            activity_factor = 1.2
        elif 16 <= current_hour < 24:  # Американская сессия
            activity_factor = 1.1  
        elif 0 <= current_hour < 8:  # Азиатская сессия
            activity_factor = 0.9
        else:
            activity_factor = 0.8  # Низкая активность
        
        return {
            "activity_factor": activity_factor,
            "session": self._get_trading_session(current_hour),
            "is_high_activity": activity_factor > 1.0
        }
    
    def _get_trading_session(self, hour: int) -> str:
        """Определение торговой сессии."""
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "european"
        elif 16 <= hour < 24:
            return "american"
        else:
            return "off_hours"
    
    def reset(self):
        """Сброс состояния симулятора."""
        self.current_condition = MarketCondition.SIDEWAYS
        self.volatility_factor = 1.0
        self.liquidity_factor = 1.0
        self.bid_ask_spread = self.spread_rate
        
        self.price_history.clear()
        self.volume_history.clear()
        self.execution_history.clear()
        
        self.last_trade_price = 0.0
        
        self.logger.debug("Симулятор рынка сброшен")