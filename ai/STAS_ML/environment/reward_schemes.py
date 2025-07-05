"""
Продвинутые схемы наград для торговой среды.
Поддерживает различные методы расчета наград и их комбинирование.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod


class BaseRewardScheme(ABC):
    """Базовый класс для схем наград."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def calculate(self, env_state: Dict) -> float:
        """Рассчитать награду на основе состояния среды."""
        pass


class ProfitReward(BaseRewardScheme):
    """Награда на основе прибыли портфеля."""
    
    def __init__(self, weight: float = 1.0, normalize: bool = True):
        super().__init__(weight)
        self.normalize = normalize
        self.last_portfolio_value = None
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state['portfolio_value']
        
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
        
        # Относительное изменение портфеля
        profit_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        if self.normalize:
            # Нормализация по волатильности
            reward = np.tanh(profit_change * 100) * self.weight
        else:
            reward = profit_change * self.weight
        
        return reward


class DrawdownPenalty(BaseRewardScheme):
    """Штраф за просадку."""
    
    def __init__(self, weight: float = -0.5, max_drawdown_threshold: float = 0.1):
        super().__init__(weight)
        self.max_drawdown_threshold = max_drawdown_threshold
    
    def calculate(self, env_state: Dict) -> float:
        max_drawdown = env_state.get('max_drawdown', 0.0)
        
        if max_drawdown > self.max_drawdown_threshold:
            # Экспоненциальный штраф за превышение порога просадки
            penalty = np.exp((max_drawdown - self.max_drawdown_threshold) * 10) - 1
            return -penalty * self.weight
        
        return 0.0


class SharpeRatioReward(BaseRewardScheme):
    """Награда на основе коэффициента Шарпа."""
    
    def __init__(self, weight: float = 0.3, window: int = 50, risk_free_rate: float = 0.02):
        super().__init__(weight)
        self.window = window
        self.risk_free_rate = risk_free_rate / 252  # Дневная безрисковая ставка
        self.returns_history = []
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < 2:
            return 0.0
        
        # Рассчитываем дневные доходности
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        self.returns_history.extend(returns[-1:])  # Добавляем только последнюю доходность
        
        # Ограничиваем размер истории
        if len(self.returns_history) > self.window:
            self.returns_history = self.returns_history[-self.window:]
        
        if len(self.returns_history) < 10:  # Минимум для расчета
            return 0.0
        
        # Рассчитываем коэффициент Шарпа
        excess_returns = np.array(self.returns_history) - self.risk_free_rate
        if np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            return np.tanh(sharpe) * self.weight
        
        return 0.0


class TradeQualityReward(BaseRewardScheme):
    """Награда за качество сделок."""
    
    def __init__(self, weight: float = 0.2, min_trades: int = 5):
        super().__init__(weight)
        self.min_trades = min_trades
    
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 0.0)
        
        if total_trades < self.min_trades:
            return 0.0
        
        # Бонус за высокую долю прибыльных сделок
        if win_rate > 0.6:
            return (win_rate - 0.5) * 2 * self.weight
        elif win_rate < 0.4:
            return -(0.5 - win_rate) * 2 * self.weight
        
        return 0.0


class VolatilityPenalty(BaseRewardScheme):
    """Штраф за высокую волатильность портфеля."""
    
    def __init__(self, weight: float = -0.1, window: int = 20):
        super().__init__(weight)
        self.window = window
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < self.window:
            return 0.0
        
        # Рассчитываем волатильность последних значений
        recent_values = portfolio_history[-self.window:]
        returns = np.diff(recent_values) / recent_values[:-1]
        volatility = np.std(returns)
        
        # Штраф за высокую волатильность
        if volatility > 0.05:  # 5% дневная волатильность
            return -volatility * 10 * self.weight
        
        return 0.0


class ConsistencyReward(BaseRewardScheme):
    """Награда за консистентность прибыли."""
    
    def __init__(self, weight: float = 0.15, window: int = 30):
        super().__init__(weight)
        self.window = window
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < self.window:
            return 0.0
        
        # Анализируем последние значения
        recent_values = portfolio_history[-self.window:]
        returns = np.diff(recent_values) / recent_values[:-1]
        
        # Доля положительных дней
        positive_days_ratio = np.sum(returns > 0) / len(returns)
        
        # Бонус за консистентность
        if positive_days_ratio > 0.6:
            return (positive_days_ratio - 0.5) * 2 * self.weight
        
        return 0.0


class MaxDrawdownPenalty(BaseRewardScheme):
    """Строгий штраф за превышение максимальной просадки."""
    
    def __init__(self, weight: float = -2.0, max_allowed_drawdown: float = 0.20):
        super().__init__(weight)
        self.max_allowed_drawdown = max_allowed_drawdown
    
    def calculate(self, env_state: Dict) -> float:
        max_drawdown = env_state.get('max_drawdown', 0.0)
        
        if max_drawdown > self.max_allowed_drawdown:
            # Экспоненциальный штраф за превышение 20% просадки
            excess = max_drawdown - self.max_allowed_drawdown
            penalty = np.exp(excess * 20) - 1  # Очень сильный штраф
            return -penalty * self.weight
        
        return 0.0


class LosingStreakPenalty(BaseRewardScheme):
    """Штраф за серию убыточных сделок подряд."""
    
    def __init__(self, weight: float = -1.0, max_losing_streak: int = 5):
        super().__init__(weight)
        self.max_losing_streak = max_losing_streak
        self.losing_streak_count = 0
        self.last_trade_count = 0
    
    def calculate(self, env_state: Dict) -> float:
        current_trade_count = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 1.0)
        
        # Отслеживаем новые сделки
        if current_trade_count > self.last_trade_count:
            # Была новая сделка, проверяем была ли она убыточной
            # Приблизительно определяем по изменению win_rate
            new_trades = current_trade_count - self.last_trade_count
            if current_trade_count > 0:
                expected_profitable = int(win_rate * current_trade_count)
                actual_profitable = int(win_rate * self.last_trade_count) if self.last_trade_count > 0 else 0
                
                if expected_profitable <= actual_profitable:
                    # Новые сделки были убыточными
                    self.losing_streak_count += new_trades
                else:
                    # Есть прибыльные сделки, сбрасываем счетчик
                    self.losing_streak_count = 0
            
            self.last_trade_count = current_trade_count
        
        # Штраф за превышение лимита убыточных сделок подряд
        if self.losing_streak_count > self.max_losing_streak:
            penalty = (self.losing_streak_count - self.max_losing_streak) ** 2
            return -penalty * self.weight
        
        return 0.0
    
    def reset(self):
        """Сброс состояния."""
        self.losing_streak_count = 0
        self.last_trade_count = 0


class PositionHoldingPenalty(BaseRewardScheme):
    """Штраф за слишком долгое держание позиции."""
    
    def __init__(self, weight: float = -1.5, max_holding_steps: int = 180):  # ~6 месяцев для дневных данных
        super().__init__(weight)
        self.max_holding_steps = max_holding_steps
        self.position_entry_step = None
        self.last_crypto_balance = 0.0
    
    def calculate(self, env_state: Dict) -> float:
        current_step = env_state.get('step', 0)
        crypto_balance = env_state.get('crypto_balance', 0.0)
        
        # Отслеживаем вход и выход из позиции
        if self.last_crypto_balance == 0 and crypto_balance > 0:
            # Вошли в позицию
            self.position_entry_step = current_step
        elif self.last_crypto_balance > 0 and crypto_balance == 0:
            # Вышли из позиции
            self.position_entry_step = None
        
        self.last_crypto_balance = crypto_balance
        
        # Штраф за слишком долгое держание
        if (self.position_entry_step is not None and 
            crypto_balance > 0 and 
            current_step - self.position_entry_step > self.max_holding_steps):
            
            excess_steps = current_step - self.position_entry_step - self.max_holding_steps
            penalty = (excess_steps / 30) ** 2  # Квадратичный рост штрафа
            return -penalty * self.weight
        
        return 0.0
    
    def reset(self):
        """Сброс состояния."""
        self.position_entry_step = None
        self.last_crypto_balance = 0.0


class EnhancedTradeQualityReward(BaseRewardScheme):
    """Улучшенная награда за качество сделок с требованием win rate > 60%."""
    
    def __init__(self, weight: float = 0.5, target_win_rate: float = 0.60, min_trades: int = 10):
        super().__init__(weight)
        self.target_win_rate = target_win_rate
        self.min_trades = min_trades
    
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 0.0)
        
        if total_trades < self.min_trades:
            return 0.0
        
        # Награда за превышение целевого win rate
        if win_rate > self.target_win_rate:
            bonus = (win_rate - self.target_win_rate) * 3  # Сильная награда
            return bonus * self.weight
        
        # Штраф за низкий win rate
        elif win_rate < self.target_win_rate:
            penalty = (self.target_win_rate - win_rate) * 2
            return -penalty * self.weight
        
        return 0.0


class TradingActivityReward(BaseRewardScheme):
    """Награда за торговую активность чтобы избежать стратегии 'ничего не делать'."""
    
    def __init__(self, weight: float = 0.1, min_trade_frequency: float = 0.01):
        super().__init__(weight)
        self.min_trade_frequency = min_trade_frequency  # Минимальная частота сделок
        self.step_count = 0
        self.last_trade_count = 0
    
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        current_step = env_state.get('step', 0)
        self.step_count = max(self.step_count, current_step)
        
        # Рассчитываем частоту торговли
        if self.step_count > 100:  # Только после 100 шагов
            trade_frequency = total_trades / self.step_count
            
            # Небольшая награда за поддержание минимальной активности
            if trade_frequency >= self.min_trade_frequency:
                return 0.1 * self.weight
            else:
                # Мягкий штраф за полное бездействие
                return -0.05 * self.weight
        
        return 0.0
    
    def reset(self):
        """Сброс состояния."""
        self.step_count = 0
        self.last_trade_count = 0


class BalancedProfitReward(BaseRewardScheme):
    """Сбалансированная награда за прибыль с учетом рисков."""
    
    def __init__(self, weight: float = 1.5, risk_adjustment: bool = True):
        super().__init__(weight)
        self.risk_adjustment = risk_adjustment
        self.last_portfolio_value = None
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state['portfolio_value']
        max_drawdown = env_state.get('max_drawdown', 0.0)
        
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
        
        # Относительное изменение портфеля
        profit_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        # Корректировка на риск
        if self.risk_adjustment and max_drawdown > 0:
            risk_factor = max(0.1, 1 - max_drawdown * 2)  # Снижаем награду при высоких просадках
            profit_change *= risk_factor
        
        # Нормализация и усиление положительных изменений
        if profit_change > 0:
            reward = np.tanh(profit_change * 100) * self.weight * 1.2  # Бонус за прибыль
        else:
            reward = np.tanh(profit_change * 100) * self.weight * 0.8  # Меньший штраф за убыток
        
        return reward


class CompositeRewardScheme:
    """Комбинированная схема наград."""
    
    def __init__(self, schemes: List[BaseRewardScheme]):
        self.schemes = schemes
        self.reward_history = []
        self.component_history = {type(scheme).__name__: [] for scheme in schemes}
    
    def calculate(self, env_state: Dict) -> float:
        """Розрахувати загальну винагороду зі збалансованим клипінгом."""
        total_reward = 0.0
        components = {}
        
        for scheme in self.schemes:
            component_reward = scheme.calculate(env_state)
            # Розширений клиппінг компонентів для більшого діапазону
            component_reward = np.clip(component_reward, -25.0, 25.0)
            total_reward += component_reward
            
            scheme_name = type(scheme).__name__
            components[scheme_name] = component_reward
            self.component_history[scheme_name].append(component_reward)
        
        # Розширений клиппінг підсумкової винагороди для кращого навчання
        total_reward = np.clip(total_reward, -100.0, 100.0)
        self.reward_history.append(total_reward)
        return total_reward
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Получить разбивку наград по компонентам."""
        breakdown = {}
        for scheme_name, history in self.component_history.items():
            if history:
                breakdown[scheme_name] = {
                    'last': history[-1],
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'total': np.sum(history)
                }
        return breakdown
    
    def reset(self):
        """Сброс истории наград."""
        self.reward_history = []
        for scheme in self.schemes:
            # Вызываем reset метод если он есть
            if hasattr(scheme, 'reset') and callable(getattr(scheme, 'reset')):
                scheme.reset()
            # Сброс общих атрибутов для схем без reset метода
            if hasattr(scheme, 'last_portfolio_value'):
                scheme.last_portfolio_value = None
            if hasattr(scheme, 'returns_history'):
                scheme.returns_history = []
        self.component_history = {type(scheme).__name__: [] for scheme in self.schemes}


def create_default_reward_scheme() -> CompositeRewardScheme:
    """Создать стандартную схему наград."""
    schemes = [
        ProfitReward(weight=1.0),
        DrawdownPenalty(weight=-0.5),
        SharpeRatioReward(weight=0.3),
        TradeQualityReward(weight=0.2),
        VolatilityPenalty(weight=-0.1),
        ConsistencyReward(weight=0.15)
    ]
    return CompositeRewardScheme(schemes)


def create_conservative_reward_scheme() -> CompositeRewardScheme:
    """Создать консервативную схему наград (акцент на стабильность)."""
    schemes = [
        ProfitReward(weight=0.7),
        DrawdownPenalty(weight=-1.0),
        SharpeRatioReward(weight=0.5),
        VolatilityPenalty(weight=-0.3),
        ConsistencyReward(weight=0.4)
    ]
    return CompositeRewardScheme(schemes)


def create_aggressive_reward_scheme() -> CompositeRewardScheme:
    """Создать агрессивную схему наград (акцент на прибыль)."""
    schemes = [
        ProfitReward(weight=1.5),
        DrawdownPenalty(weight=-0.2),
        TradeQualityReward(weight=0.3),
        SharpeRatioReward(weight=0.2)
    ]
    return CompositeRewardScheme(schemes)


def create_optimized_reward_scheme() -> CompositeRewardScheme:
    """
    Створити збалансовану схему винагород з різноманітними значеннями:
    - Основна винагорода за прибуток з підвищеним вагом
    - Гнучкий контроль ризиків
    - Стимулювання якісних торгових рішень
    - Запобігання екстремальним просадкам
    - Збалансований діапазон винагород для кращого навчання
    """
    schemes = [
        # Основна винагорода за прибуток (підвищений вес)
        BalancedProfitReward(weight=3.0, risk_adjustment=True),
        
        # Помірний контроль просадки з більшим вагом
        MaxDrawdownPenalty(weight=-4.0, max_allowed_drawdown=0.20),
        
        # Винагорода за якість торгівлі
        EnhancedTradeQualityReward(weight=2.0, target_win_rate=0.55, min_trades=15),
        
        # Штраф за серії збиткових угод
        LosingStreakPenalty(weight=-2.5, max_losing_streak=6),
        
        # Контроль часу утримання позиції
        PositionHoldingPenalty(weight=-1.5, max_holding_steps=150),
        
        # Стимулювання торгової активності
        TradingActivityReward(weight=1.0, min_trade_frequency=0.015),
        
        # Додаткові метрики зі збільшеними вагами
        SharpeRatioReward(weight=1.5, window=40),
        VolatilityPenalty(weight=-0.8),
        ConsistencyReward(weight=1.2, window=25)
    ]
    return CompositeRewardScheme(schemes)


class TradingMetrics:
    """Расчет торговых метрик."""
    
    @staticmethod
    def calculate_all_metrics(portfolio_history: List[float], 
                            trade_history: List[Dict],
                            initial_balance: float,
                            risk_free_rate: float = 0.02) -> Dict:
        """Рассчитать все торговые метрики."""
        if not portfolio_history:
            return {}
        
        portfolio_array = np.array(portfolio_history)
        
        # Основные метрики
        total_return = (portfolio_array[-1] - initial_balance) / initial_balance
        
        # Просадки
        peaks = np.maximum.accumulate(portfolio_array)
        drawdowns = (peaks - portfolio_array) / peaks
        max_drawdown = np.max(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns > 0]) if np.any(drawdowns > 0) else 0.0
        
        # Доходности
        returns = np.diff(portfolio_array) / portfolio_array[:-1]
        
        # Коэффициент Шарпа
        if len(returns) > 1 and np.std(returns) > 0:
            excess_returns = returns - risk_free_rate / 252
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
        else:
            sharpe_ratio = 0.0
        
        # Коэффициент Сортино
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.sqrt(np.mean(negative_returns**2))
            sortino_ratio = np.sqrt(252) * np.mean(returns) / downside_std
        else:
            sortino_ratio = sharpe_ratio
        
        # Месячные доходности
        if len(portfolio_array) >= 30:
            monthly_returns = []
            for i in range(30, len(portfolio_array), 30):
                start_val = portfolio_array[i-30]
                end_val = portfolio_array[i]
                if start_val > 0:
                    monthly_return = (end_val - start_val) / start_val
                    monthly_returns.append(monthly_return)
            
            avg_monthly_return = np.mean(monthly_returns) if monthly_returns else 0.0
            monthly_volatility = np.std(monthly_returns) if len(monthly_returns) > 1 else 0.0
        else:
            avg_monthly_return = 0.0
            monthly_volatility = 0.0
        
        # Метрики сделок
        if trade_history:
            profitable_trades = sum(1 for trade in trade_history 
                                  if trade.get('profit', 0) > 0)
            win_rate = profitable_trades / len(trade_history)
            
            profits = [trade.get('profit', 0) for trade in trade_history if trade.get('profit', 0) > 0]
            losses = [abs(trade.get('profit', 0)) for trade in trade_history if trade.get('profit', 0) < 0]
            
            avg_profit = np.mean(profits) if profits else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            profit_factor = sum(profits) / sum(losses) if losses and sum(losses) > 0 else 0.0
        else:
            win_rate = 0.0
            avg_profit = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        # Волатильность
        annual_volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # Calmar Ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0
        
        return {
            'total_return': total_return,
            'annual_return': total_return * (252 / len(portfolio_array)) if len(portfolio_array) > 0 else 0.0,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annual_volatility': annual_volatility,
            'avg_monthly_return': avg_monthly_return,
            'monthly_volatility': monthly_volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'total_trades': len(trade_history),
            'profitable_trades': sum(1 for trade in trade_history if trade.get('profit', 0) > 0)
        } 