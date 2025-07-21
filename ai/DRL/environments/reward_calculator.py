"""Калькулятор наград для торговой среды."""

import numpy as np
from typing import Dict, List, Optional, Any
from enum import Enum

from ..config import TradingConfig
from ..utils import DRLLogger, TradingMetrics


class RewardScheme(Enum):
    """Схемы расчета наград."""
    PROFIT_BASED = "profit_based"
    SHARPE_BASED = "sharpe_based"
    RISK_ADJUSTED = "risk_adjusted"
    DIFFERENTIAL = "differential"
    MULTI_OBJECTIVE = "multi_objective"


class RewardCalculator:
    """
    Калькулятор наград для торговой среды.
    
    Реализует различные схемы расчета наград для обучения DRL агентов,
    включая основанные на прибыли, риск-скорректированные и комплексные подходы.
    """
    
    def __init__(self, config: TradingConfig, logger: Optional[DRLLogger] = None):
        """
        Инициализация калькулятора наград.
        
        Args:
            config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        self.config = config
        self.logger = logger or DRLLogger("reward_calculator")
        
        # Схема расчета наград
        self.scheme = RewardScheme(config.reward_scheme)
        
        # История для расчета метрик
        self.portfolio_history: List[float] = []
        self.return_history: List[float] = []
        self.action_history: List[str] = []
        
        # Кэш для оптимизации
        self.benchmark_return = 0.0  # Buy & Hold доходность
        self.episode_start_value = 0.0
        
        # Параметры наград
        self.reward_scaling = config.reward_scaling
        self.penalty_for_inaction = config.penalty_for_inaction
        
        # Технические параметры
        self.lookback_window = min(50, config.lookback_window)  # Окно для расчета метрик
        
        self.logger.info(f"Калькулятор наград инициализирован со схемой: {self.scheme.value}")
    
    def reset(self):
        """Сброс состояния калькулятора."""
        self.portfolio_history.clear()
        self.return_history.clear()
        self.action_history.clear()
        self.benchmark_return = 0.0
        self.episode_start_value = 0.0
        
        self.logger.debug("Калькулятор наград сброшен")
    
    def calculate_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        trade_info: Dict[str, Any],
        current_step: int,
        market_data: Dict[str, float]
    ) -> float:
        """
        Расчет награды за выполненное действие.
        
        Args:
            prev_portfolio_value: предыдущая стоимость портфеля
            current_portfolio_value: текущая стоимость портфеля
            trade_info: информация о выполненной сделке
            current_step: текущий шаг
            market_data: текущие рыночные данные
            
        Returns:
            Рассчитанная награда
        """
        # Обновление истории
        self._update_history(current_portfolio_value, trade_info)
        
        # Базовый расчет награды по схеме
        if self.scheme == RewardScheme.PROFIT_BASED:
            reward = self._calculate_profit_based_reward(prev_portfolio_value, current_portfolio_value, trade_info)
        elif self.scheme == RewardScheme.SHARPE_BASED:
            reward = self._calculate_sharpe_based_reward(prev_portfolio_value, current_portfolio_value)
        elif self.scheme == RewardScheme.RISK_ADJUSTED:
            reward = self._calculate_risk_adjusted_reward(prev_portfolio_value, current_portfolio_value, trade_info)
        elif self.scheme == RewardScheme.DIFFERENTIAL:
            reward = self._calculate_differential_reward(current_portfolio_value, market_data)
        elif self.scheme == RewardScheme.MULTI_OBJECTIVE:
            reward = self._calculate_multi_objective_reward(prev_portfolio_value, current_portfolio_value, trade_info, market_data)
        else:
            reward = self._calculate_profit_based_reward(prev_portfolio_value, current_portfolio_value, trade_info)
        
        # Применение дополнительных модификаторов
        reward = self._apply_reward_modifiers(reward, trade_info, current_step, market_data)
        
        # Масштабирование награды
        final_reward = reward * self.reward_scaling
        
        self.logger.debug(f"Награда рассчитана: базовая={reward:.6f}, финальная={final_reward:.6f}")
        
        return final_reward
    
    def _calculate_profit_based_reward(
        self, 
        prev_value: float, 
        current_value: float, 
        trade_info: Dict[str, Any]
    ) -> float:
        """Расчет награды на основе прибыли."""
        # Базовая награда - относительное изменение стоимости портфеля
        if prev_value <= 0:
            return 0.0
        
        portfolio_return = (current_value - prev_value) / prev_value
        
        # Основная награда
        reward = portfolio_return
        
        # Бонус за выполнение сделки (если она прибыльная)
        if trade_info.get("executed", False) and trade_info.get("pnl", 0) > 0:
            trade_bonus = min(trade_info["pnl"] / prev_value, 0.01)  # Максимум 1%
            reward += trade_bonus
        
        # Штраф за комиссии
        if trade_info.get("commission", 0) > 0:
            commission_penalty = trade_info["commission"] / prev_value
            reward -= commission_penalty
        
        return reward
    
    def _calculate_sharpe_based_reward(self, prev_value: float, current_value: float) -> float:
        """Расчет награды на основе коэффициента Шарпа."""
        if prev_value <= 0:
            return 0.0
        
        # Добавление текущего возврата
        current_return = (current_value - prev_value) / prev_value
        self.return_history.append(current_return)
        
        # Расчет Sharpe ratio для окна
        if len(self.return_history) < 10:
            return current_return  # Недостаточно данных для Sharpe
        
        window_returns = self.return_history[-self.lookback_window:]
        
        try:
            sharpe_ratio = TradingMetrics.sharpe_ratio(window_returns)
            # Нормализация Sharpe ratio в диапазон [-1, 1]
            normalized_sharpe = np.tanh(sharpe_ratio / 2.0)
            
            # Комбинация текущего возврата и Sharpe ratio
            reward = 0.7 * current_return + 0.3 * normalized_sharpe
            
        except (ZeroDivisionError, ValueError):
            reward = current_return
        
        return reward
    
    def _calculate_risk_adjusted_reward(
        self, 
        prev_value: float, 
        current_value: float, 
        trade_info: Dict[str, Any]
    ) -> float:
        """Расчет риск-скорректированной награды."""
        if prev_value <= 0:
            return 0.0
        
        # Базовая прибыль
        portfolio_return = (current_value - prev_value) / prev_value
        
        # Штраф за волатильность
        if len(self.return_history) >= 5:
            recent_returns = self.return_history[-5:]
            volatility = np.std(recent_returns)
            volatility_penalty = min(volatility * 2, 0.01)  # Максимальный штраф 1%
        else:
            volatility_penalty = 0.0
        
        # Штраф за просадку
        if hasattr(self, 'portfolio_history') and len(self.portfolio_history) >= 2:
            peak_value = max(self.portfolio_history[-self.lookback_window:])
            drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
            drawdown_penalty = drawdown * 0.5  # 50% от просадки
        else:
            drawdown_penalty = 0.0
        
        # Бонус за управление рисками
        position_size = abs(trade_info.get("position_size", 0))
        if position_size > 0.8:  # Слишком большая позиция
            risk_penalty = (position_size - 0.8) * 0.1
        else:
            risk_penalty = 0.0
        
        reward = portfolio_return - volatility_penalty - drawdown_penalty - risk_penalty
        
        return reward
    
    def _calculate_differential_reward(self, current_value: float, market_data: Dict[str, float]) -> float:
        """Расчет дифференциальной награды (превышение над бенчмарком)."""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Расчет доходности портфеля
        prev_value = self.portfolio_history[-2]
        if prev_value <= 0:
            return 0.0
        
        portfolio_return = (current_value - prev_value) / prev_value
        
        # Расчет бенчмарка (Buy & Hold)
        if len(self.portfolio_history) >= 2:
            # Предполагаем, что у нас есть данные о ценах
            current_price = market_data.get("close", 0)
            if hasattr(self, '_benchmark_price') and self._benchmark_price > 0:
                benchmark_return = (current_price - self._benchmark_price) / self._benchmark_price
                self._benchmark_price = current_price
            else:
                self._benchmark_price = current_price
                benchmark_return = 0.0
        else:
            benchmark_return = 0.0
        
        # Награда = превышение над бенчмарком
        differential_return = portfolio_return - benchmark_return
        
        return differential_return
    
    def _calculate_multi_objective_reward(
        self,
        prev_value: float,
        current_value: float,
        trade_info: Dict[str, Any],
        market_data: Dict[str, float]
    ) -> float:
        """Расчет мультицелевой награды."""
        if prev_value <= 0:
            return 0.0
        
        # Компоненты награды с весами
        weights = {
            "profit": 0.4,      # Прибыль
            "risk": 0.2,        # Управление рисками
            "sharpe": 0.2,      # Качество доходности
            "efficiency": 0.1,  # Эффективность торговли
            "consistency": 0.1  # Постоянство результатов
        }
        
        # 1. Компонент прибыли
        profit_component = (current_value - prev_value) / prev_value
        
        # 2. Компонент риска
        risk_component = self._calculate_risk_component(trade_info)
        
        # 3. Компонент Sharpe
        sharpe_component = self._calculate_sharpe_component()
        
        # 4. Компонент эффективности торговли
        efficiency_component = self._calculate_efficiency_component(trade_info)
        
        # 5. Компонент постоянства
        consistency_component = self._calculate_consistency_component()
        
        # Взвешенная сумма
        reward = (
            weights["profit"] * profit_component +
            weights["risk"] * risk_component +
            weights["sharpe"] * sharpe_component +
            weights["efficiency"] * efficiency_component +
            weights["consistency"] * consistency_component
        )
        
        return reward
    
    def _calculate_risk_component(self, trade_info: Dict[str, Any]) -> float:
        """Расчет риск-компонента награды."""
        risk_score = 0.0
        
        # Штраф за большой размер позиции
        position_size = abs(trade_info.get("position_size", 0))
        if position_size > self.config.max_position_size:
            risk_score -= (position_size - self.config.max_position_size) * 2
        
        # Бонус за диверсификацию (в данном случае за не максимальную позицию)
        if 0.2 <= position_size <= 0.8:
            risk_score += 0.01
        
        # Штраф за чрезмерную торговлю
        if len(self.action_history) >= 10:
            recent_actions = self.action_history[-10:]
            trade_frequency = sum(1 for action in recent_actions if action in ["buy", "sell"]) / len(recent_actions)
            if trade_frequency > 0.8:  # Более 80% торговых действий
                risk_score -= 0.02
        
        return risk_score
    
    def _calculate_sharpe_component(self) -> float:
        """Расчет Sharpe-компонента награды."""
        if len(self.return_history) < 10:
            return 0.0
        
        try:
            recent_returns = self.return_history[-self.lookback_window:]
            sharpe_ratio = TradingMetrics.sharpe_ratio(recent_returns)
            
            # Нормализация в диапазон [-0.05, 0.05]
            normalized_sharpe = np.tanh(sharpe_ratio) * 0.05
            
            return normalized_sharpe
        except:
            return 0.0
    
    def _calculate_efficiency_component(self, trade_info: Dict[str, Any]) -> float:
        """Расчет компонента эффективности торговли."""
        efficiency_score = 0.0
        
        # Штраф за комиссии
        if trade_info.get("commission", 0) > 0:
            commission_ratio = trade_info["commission"] / trade_info.get("value", 1)
            efficiency_score -= commission_ratio
        
        # Бонус за прибыльную сделку
        if trade_info.get("executed", False) and trade_info.get("pnl", 0) > 0:
            pnl_ratio = trade_info["pnl"] / trade_info.get("value", 1)
            efficiency_score += min(pnl_ratio, 0.02)  # Максимум 2%
        
        # Штраф за бездействие (если настроено)
        if not trade_info.get("executed", False) and self.penalty_for_inaction > 0:
            efficiency_score -= self.penalty_for_inaction
        
        return efficiency_score
    
    def _calculate_consistency_component(self) -> float:
        """Расчет компонента постоянства результатов."""
        if len(self.return_history) < 5:
            return 0.0
        
        recent_returns = self.return_history[-10:]
        
        # Оценка стабильности через коэффициент вариации
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if mean_return != 0:
            cv = abs(std_return / mean_return)  # Коэффициент вариации
            # Бонус за низкую вариацию (высокую стабильность)
            consistency_score = max(0, 0.02 - cv * 0.1)
        else:
            consistency_score = 0.0
        
        return consistency_score
    
    def _apply_reward_modifiers(
        self,
        base_reward: float,
        trade_info: Dict[str, Any],
        current_step: int,
        market_data: Dict[str, float]
    ) -> float:
        """Применение дополнительных модификаторов к награде."""
        modified_reward = base_reward
        
        # Модификатор времени (поощрение долгосрочных стратегий)
        if current_step > 100:  # После начального периода
            time_bonus = min(current_step / 1000, 0.001)
            modified_reward += time_bonus
        
        # Модификатор за достижение целей
        if hasattr(self, 'portfolio_history') and len(self.portfolio_history) >= 2:
            current_value = self.portfolio_history[-1]
            if self.episode_start_value > 0:
                total_return = (current_value - self.episode_start_value) / self.episode_start_value
                
                # Бонус за достижение месячной цели
                if total_return >= self.config.target_monthly_return:
                    modified_reward += 0.1  # Значительный бонус
                elif total_return >= self.config.target_monthly_return * 0.5:
                    modified_reward += 0.02  # Малый бонус за прогресс
        
        # Штраф за чрезмерные потери
        if base_reward < -0.05:  # Потери более 5%
            loss_penalty = abs(base_reward) * 0.5  # Дополнительный штраф
            modified_reward -= loss_penalty
        
        # Ограничение диапазона награды
        modified_reward = np.clip(modified_reward, -1.0, 1.0)
        
        return modified_reward
    
    def _update_history(self, portfolio_value: float, trade_info: Dict[str, Any]):
        """Обновление истории для расчета метрик."""
        # Обновление истории портфеля
        self.portfolio_history.append(portfolio_value)
        if len(self.portfolio_history) > self.lookback_window * 2:
            self.portfolio_history = self.portfolio_history[-self.lookback_window:]
        
        # Обновление истории действий
        action = trade_info.get("action", "hold")
        self.action_history.append(action)
        if len(self.action_history) > self.lookback_window:
            self.action_history = self.action_history[-self.lookback_window:]
        
        # Установка начального значения эпизода
        if self.episode_start_value == 0.0:
            self.episode_start_value = portfolio_value
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Получение статистики по наградам."""
        if len(self.return_history) < 2:
            return {"total_returns": 0}
        
        returns = np.array(self.return_history)
        
        stats = {
            "total_returns": len(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "positive_returns": int(np.sum(returns > 0)),
            "negative_returns": int(np.sum(returns < 0)),
            "win_rate": float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
        }
        
        # Sharpe ratio если достаточно данных
        if len(returns) >= 10:
            try:
                stats["sharpe_ratio"] = TradingMetrics.sharpe_ratio(returns)
            except:
                stats["sharpe_ratio"] = 0.0
        
        return stats
    
    def get_current_performance(self) -> Dict[str, float]:
        """Получение текущих показателей производительности."""
        if len(self.portfolio_history) < 2:
            return {"portfolio_return": 0.0}
        
        current_value = self.portfolio_history[-1]
        start_value = self.episode_start_value if self.episode_start_value > 0 else self.portfolio_history[0]
        
        portfolio_return = (current_value - start_value) / start_value if start_value > 0 else 0.0
        
        performance = {
            "portfolio_return": portfolio_return,
            "current_value": current_value,
            "start_value": start_value,
            "steps_taken": len(self.portfolio_history),
            "trades_executed": sum(1 for action in self.action_history if action in ["buy", "sell"])
        }
        
        # Добавление целевых метрик
        performance["target_return"] = self.config.target_monthly_return
        performance["target_progress"] = portfolio_return / self.config.target_monthly_return if self.config.target_monthly_return > 0 else 0
        
        return performance