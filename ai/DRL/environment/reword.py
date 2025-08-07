"""
Модуль системы вознаграждений для DRL торговой среды
Содержит продвинутые схемы вознаграждений на основе современных исследований в FinRL
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """Стороны ордера"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

@dataclass
class Trade:
    """Информация о сделке"""
    timestamp: int
    price: float
    quantity: float
    side: OrderSide
    order_type: OrderType
    commission: float
    slippage: float
    market_impact: float
    realized_pnl: float = 0.0

@dataclass
class Position:
    """Информация о позиции"""
    symbol: str
    size: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    last_update: int

class BaseRewardScheme:
    """Базовый класс для схем вознаграждений"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def calculate_reward(self, **kwargs) -> float:
        """Базовый метод расчета вознаграждения"""
        raise NotImplementedError("Subclasses must implement calculate_reward")

class SimpleRewardScheme(BaseRewardScheme):
    """Простая схема вознаграждений - базовая доходность"""
    
    def calculate_reward(self, 
                        current_portfolio_value: float,
                        previous_portfolio_value: float,
                        **kwargs) -> float:
        """Простое вознаграждение на основе изменения портфеля"""
        return (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value * 100

class AdvancedRewardScheme(BaseRewardScheme):
    """Продвинутая схема вознаграждений на основе современных исследований в FinRL"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.returns_history = deque(maxlen=252)  # 1 год дневных данных
        self.drawdown_history = deque(maxlen=100)
        self.sharpe_window = deque(maxlen=50)
        
        # Новые метрики для более реалистичного вознаграждения
        self.trade_count = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        
        # Адаптивные параметры
        self.risk_tolerance = config.get('risk_tolerance', 0.02)
        self.transaction_cost_penalty = config.get('transaction_cost_penalty', 1.0)
        self.stability_bonus = config.get('stability_bonus', 0.1)
        
        logger.info("Инициализирована продвинутая схема вознаграждений с адаптивными параметрами")
    
    def calculate_advanced_sharpe_ratio(self, returns: np.ndarray, window: int = 30) -> float:
        """Расчет продвинутого Sharpe ratio с учетом хвостовых рисков"""
        if len(returns) < window:
            return 0.0
            
        recent_returns = returns[-window:]
        mean_return = np.mean(recent_returns)
        
        # Используем модифицированный VaR вместо стандартного отклонения
        var_95 = np.percentile(recent_returns, 5)  # 5% VaR
        if var_95 >= 0:
            return mean_return  # Если нет отрицательных доходностей
        
        modified_sharpe = mean_return / abs(var_95)
        return modified_sharpe
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Расчет Calmar ratio (годовая доходность / максимальная просадка)"""
        if len(returns) < 30:
            return 0.0
            
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        if max_drawdown == 0:
            return np.mean(returns) * 252  # Годовая доходность
        
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    def calculate_information_ratio(self, returns: np.ndarray, market_return: float, window: int = 20) -> float:
        """Расчет информационного коэффициента"""
        if len(returns) < window:
            return 0.0
            
        returns_list = list(returns)
        excess_returns = np.array(returns_list) - market_return
        tracking_error = np.std(excess_returns[-window:])
        information_ratio = np.mean(excess_returns[-window:]) / max(tracking_error, 1e-6)
        return information_ratio
        
    def calculate_reward(self, 
                        current_portfolio_value: float,
                        previous_portfolio_value: float,
                        current_position: Position,
                        market_return: float,
                        volatility: float,
                        trade_info: Optional[Trade] = None) -> float:
        """
        Многокомпонентная система вознаграждений
        Основана на лучших практиках из FinRL, современных исследований и финансовой теории
        """
        # 1. Базовое вознаграждение от доходности
        portfolio_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
        self.returns_history.append(portfolio_return)
        
        reward = 0.0
        
        # 2. Reward shaping для стабильности обучения
        raw_return_reward = portfolio_return * 100  # Масштабирование для лучшей обучаемости
        
        # 3. Продвинутый Risk-adjusted return
        if len(self.returns_history) > 10:
            returns_array = np.array(list(self.returns_history))
            
            # Используем продвинутый Sharpe ratio с VaR
            advanced_sharpe = self.calculate_advanced_sharpe_ratio(returns_array)
            self.sharpe_window.append(advanced_sharpe)
            sharpe_reward = advanced_sharpe * 0.15
            
            # Добавляем Calmar ratio для учета максимальных просадок
            calmar_ratio = self.calculate_calmar_ratio(returns_array)
            calmar_reward = calmar_ratio * 0.05
        else:
            sharpe_reward = 0.0
            calmar_reward = 0.0
        
        # 4. Динамический Drawdown penalty
        if len(self.returns_history) > 5:
            cumulative_returns = np.cumprod(1 + np.array(list(self.returns_history)))
            running_max = np.maximum.accumulate(cumulative_returns)
            current_drawdown = (running_max[-1] - cumulative_returns[-1]) / running_max[-1]
            
            # Прогрессивный штраф за просадки
            if current_drawdown > 0.15:  # Критическая просадка
                drawdown_penalty = -current_drawdown * 100
            elif current_drawdown > 0.08:  # Значительная просадка
                drawdown_penalty = -current_drawdown * 60
            elif current_drawdown > 0.03:  # Умеренная просадка
                drawdown_penalty = -current_drawdown * 30
            else:
                drawdown_penalty = 0
        else:
            drawdown_penalty = 0
        
        # 5. Продвинутая оценка качества сделок
        trade_reward = 0.0
        if trade_info:
            self.trade_count += 1
            self.total_commission += trade_info.commission
            
            if trade_info.realized_pnl > 0:
                self.winning_trades += 1
                self.current_consecutive_losses = 0
                # Бонус за прибыльные сделки с учетом риска
                risk_adjusted_profit = trade_info.realized_pnl / max(volatility, 0.01)
                trade_reward += risk_adjusted_profit * 15
            else:
                self.current_consecutive_losses += 1
                self.max_consecutive_losses = max(self.max_consecutive_losses, 
                                                self.current_consecutive_losses)
                # Штраф за убыточные сделки
                trade_reward += trade_info.realized_pnl * 8
                
                # Дополнительный штраф за серию убытков
                if self.current_consecutive_losses > 3:
                    consecutive_penalty = -(self.current_consecutive_losses - 3) * 5
                    trade_reward += consecutive_penalty
                
            # Адаптивный штраф за комиссии
            commission_penalty = -trade_info.commission * self.transaction_cost_penalty * 25
            trade_reward += commission_penalty
        
        # 6. Продвинутый market outperformance с информационным коэффициентом
        alpha = portfolio_return - market_return
        
        # Информационный коэффициент (активный доход / трекинг эррор)
        if len(self.returns_history) > 20:
            information_ratio = self.calculate_information_ratio(
                np.array(list(self.returns_history)), market_return
            )
            alpha_reward = information_ratio * 10
        else:
            alpha_reward = alpha * 15 if alpha > 0 else alpha * 8
        
        # 7. Адаптивный volatility penalty с учетом режима рынка
        if len(self.returns_history) > 10:
            returns_list = list(self.returns_history)
            recent_vol = np.std(returns_list[-10:])
            market_vol = volatility
            
            # Штраф за избыточную волатильность относительно рынка
            excess_vol = recent_vol - market_vol
            if excess_vol > self.risk_tolerance:
                vol_penalty = -excess_vol * 40
            else:
                vol_penalty = 0
        else:
            vol_penalty = 0.0
        
        # 8. Улучшенный position sizing с динамическим лимитом
        if current_portfolio_value > 0:
            position_size_ratio = abs(current_position.size * current_position.avg_price) / current_portfolio_value
            
            # Динамический лимит в зависимости от волатильности
            max_position_ratio = 0.9 - (volatility * 2)  # Меньше leverage при высокой волатильности
            max_position_ratio = np.clip(max_position_ratio, 0.3, 0.95)
            
            if position_size_ratio > max_position_ratio:
                position_penalty = -(position_size_ratio - max_position_ratio) * 120
            else:
                position_penalty = 0.0
        else:
            position_penalty = 0.0
        
        # 9. Бонус за стабильность (новый компонент)
        stability_bonus = 0.0
        if len(self.returns_history) > 30:
            returns_list = list(self.returns_history)
            recent_returns = np.array(returns_list[-30:])
            stability_metric = 1.0 / (1.0 + np.std(recent_returns))
            win_rate = self.winning_trades / max(self.trade_count, 1)
            
            if win_rate > 0.6 and stability_metric > 0.8:
                stability_bonus = self.stability_bonus * 10
        
        # Комбинируем все компоненты
        reward = (raw_return_reward + 
                 sharpe_reward + 
                 calmar_reward +
                 drawdown_penalty + 
                 trade_reward + 
                 alpha_reward + 
                 vol_penalty + 
                 position_penalty +
                 stability_bonus)
        
        # Адаптивный клампинг в зависимости от волатильности
        max_reward = 15.0 if volatility < 0.02 else 12.0
        min_reward = -15.0 if volatility < 0.02 else -12.0
        reward = np.clip(reward, min_reward, max_reward)
        
        return reward
    
    def get_metrics(self) -> Dict[str, float]:
        """Получение текущих метрик системы вознаграждений"""
        return {
            'total_trades': self.trade_count,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.trade_count, 1),
            'total_commission': self.total_commission,
            'max_consecutive_losses': self.max_consecutive_losses,
            'current_consecutive_losses': self.current_consecutive_losses,
            'returns_history_length': len(self.returns_history),
            'average_return': np.mean(list(self.returns_history)) if self.returns_history else 0.0,
            'return_volatility': np.std(list(self.returns_history)) if len(self.returns_history) > 1 else 0.0
        }
    
    def reset_metrics(self):
        """Сброс всех накопленных метрик"""
        self.returns_history.clear()
        self.drawdown_history.clear() 
        self.sharpe_window.clear()
        self.trade_count = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.max_consecutive_losses = 0
        self.current_consecutive_losses = 0
        logger.info("Метрики системы вознаграждений сброшены")

class CurriculumRewardScheme(AdvancedRewardScheme):
    """Схема вознаграждений для curriculum learning с адаптивными параметрами"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.learning_stage = config.get('learning_stage', 'bitcoin')  # bitcoin, ethereum, mixed
        self.stage_multipliers = {
            'bitcoin': {'stability': 1.5, 'risk_penalty': 0.8, 'volatility_tolerance': 1.2},
            'ethereum': {'stability': 1.0, 'risk_penalty': 1.2, 'volatility_tolerance': 0.8},
            'mixed': {'stability': 1.2, 'risk_penalty': 1.0, 'volatility_tolerance': 1.0}
        }
        
    def calculate_reward(self, **kwargs) -> float:
        """Адаптивное вознаграждение в зависимости от этапа обучения"""
        base_reward = super().calculate_reward(**kwargs)
        
        # Применяем множители в зависимости от этапа
        multipliers = self.stage_multipliers.get(self.learning_stage, self.stage_multipliers['mixed'])
        
        # Модифицируем награду в зависимости от этапа обучения
        if self.learning_stage == 'bitcoin':
            # На этапе Bitcoin поощряем стабильность
            base_reward *= multipliers['stability']
        elif self.learning_stage == 'ethereum':
            # На этапе Ethereum учитываем высокую волатильность
            base_reward *= multipliers['volatility_tolerance']
        else:  # mixed
            # На смешанном этапе балансируем все факторы
            base_reward *= multipliers['risk_penalty']
            
        return base_reward

def create_reward_scheme(scheme_type: str = "advanced", config: Dict[str, Any] = None) -> BaseRewardScheme:
    """
    Фабричная функция для создания схемы вознаграждений
    
    Args:
        scheme_type: тип схемы ('simple', 'advanced', 'curriculum')
        config: конфигурация для схемы
        
    Returns:
        Экземпляр схемы вознаграждений
    """
    if config is None:
        config = {}
        
    if scheme_type == "simple":
        return SimpleRewardScheme(config)
    elif scheme_type == "advanced":
        return AdvancedRewardScheme(config)
    elif scheme_type == "curriculum":
        return CurriculumRewardScheme(config)
    else:
        raise ValueError(f"Unknown reward scheme type: {scheme_type}")

# Алиасы для обратной совместимости
RewardScheme = AdvancedRewardScheme