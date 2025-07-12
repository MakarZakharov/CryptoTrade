"""
Продвинуті схеми винагород для торгового середовища.
Підтримує різні методи розрахунку винагород та їх комбінування.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Callable
from abc import ABC, abstractmethod


class BaseRewardScheme(ABC):
    """Базовий клас для схем винагород."""
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    @abstractmethod
    def calculate(self, env_state: Dict) -> float:
        """Розрахувати винагороду на основі стану середовища."""
        pass
    
    def reset(self):
        """Скинути стан схеми винагород."""
        pass


class ProfitReward(BaseRewardScheme):
    """Винагорода на основі прибутку портфеля."""
    
    def __init__(self, weight: float = 1.0, normalize: bool = True):
        super().__init__(weight)
        self.normalize = normalize
        self.last_portfolio_value = None
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state['portfolio_value']
        
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
        
        # Відносна зміна портфеля
        profit_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        if self.normalize:
            # Нормалізація по волатильності
            reward = np.tanh(profit_change * 100) * self.weight
        else:
            reward = profit_change * self.weight
        
        return reward
    
    def reset(self):
        self.last_portfolio_value = None


class DrawdownPenalty(BaseRewardScheme):
    """Штраф за просадку."""
    
    def __init__(self, weight: float = -0.5, max_drawdown_threshold: float = 0.1):
        super().__init__(weight)
        self.max_drawdown_threshold = max_drawdown_threshold
    
    def calculate(self, env_state: Dict) -> float:
        max_drawdown = env_state.get('max_drawdown', 0.0)
        
        if max_drawdown > self.max_drawdown_threshold:
            # Експоненційний штраф за перевищення порогу просадки
            penalty = np.exp((max_drawdown - self.max_drawdown_threshold) * 10) - 1
            return -penalty * abs(self.weight)
        
        return 0.0


class DynamicRewardScaler:
    """
    Динамічне масштабування винагород на основі історичних результатів.
    Запобігає завеликим винагородам через адаптивну нормалізацію.
    """
    
    def __init__(self, history_window: int = 100, target_range: tuple = (-2.0, 2.0)):
        self.history_window = history_window
        self.target_range = target_range
        self.reward_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
        self.alpha = 0.1  # Коефіцієнт згладжування для експоненційного середнього
    
    def scale_reward(self, raw_reward: float) -> float:
        """Масштабує винагороду на основі історичних даних."""
        # Додаємо поточну винагороду до історії
        self.reward_history.append(raw_reward)
        
        # Обмежуємо розмір історії
        if len(self.reward_history) > self.history_window:
            self.reward_history = self.reward_history[-self.history_window:]
        
        # Оновлюємо статистики експоненційним згладжуванням
        if len(self.reward_history) > 1:
            current_mean = np.mean(self.reward_history[-10:])  # Останні 10 значень
            current_std = np.std(self.reward_history[-10:]) + 1e-8  # Запобігаємо діленню на 0
            
            # Експоненційне згладжування
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * current_mean
            self.running_std = (1 - self.alpha) * self.running_std + self.alpha * current_std
        
        # Z-score нормалізація з адаптивними параметрами
        if self.running_std > 0:
            normalized_reward = (raw_reward - self.running_mean) / self.running_std
        else:
            normalized_reward = 0.0
        
        # Масштабування до цільового діапазону з tanh для м'якого обмеження
        target_min, target_max = self.target_range
        target_center = (target_max + target_min) / 2
        target_scale = (target_max - target_min) / 4  # tanh(±2) ≈ ±0.96
        
        scaled_reward = target_center + target_scale * np.tanh(normalized_reward)
        
        # Логування для діагностики (тільки при великих відхиленнях)
        if abs(raw_reward) > 10:
            print(f"🔧 REWARD SCALING: {raw_reward:.2f} -> {scaled_reward:.2f} (μ={self.running_mean:.2f}, σ={self.running_std:.2f})")
        
        return scaled_reward
    
    def reset(self):
        """Скидання статистик (частково зберігаємо історію для стабільності)."""
        # Зберігаємо половину історії для кращої стабільності
        if len(self.reward_history) > 20:
            self.reward_history = self.reward_history[-20:]
        # Не скидаємо running_mean і running_std повністю для стабільності


class CompositeRewardScheme:
    """Комбінована схема винагород з динамічним масштабуванням."""
    
    def __init__(self, schemes: List[BaseRewardScheme], enable_dynamic_scaling: bool = True):
        self.schemes = schemes
        self.reward_history = []
        self.enable_dynamic_scaling = enable_dynamic_scaling
        
        # Ініціалізуємо динамічний масштабувач
        if self.enable_dynamic_scaling:
            self.scaler = DynamicRewardScaler(
                history_window=50,  # Менше вікно для швидшої адаптації
                target_range=(-3.0, 3.0)  # Розумний діапазон для PPO
            )
        
        # СИСТЕМА ПРОГРЕСИВНИХ НАГОРОД ВИМКНЕНА через некоректні порівняння
        self.last_performance = {
            'return': 0.0,
            'drawdown': 0.0,
            'trades': 0,
            'win_rate': 0.0
        }
        self.improvement_bonus_count = 0
        
        # СИСТЕМА ЕСКАЛАЦІЇ ШТРАФІВ ЗА ПОСЛІДОВНУ ПОГАНУ ПРОДУКТИВНІСТЬ
        self.consecutive_poor_episodes = 0
        self.poor_performance_threshold = -0.15  # -15% як поганий результат (збільшено з -2%)
    
    def calculate(self, env_state: Dict) -> float:
        """Розрахувати загальну винагороду з динамічним масштабуванням та контролем від'ємної продуктивності."""
        total_reward = 0.0
        component_rewards = {}
        
        for scheme in self.schemes:
            component_reward = scheme.calculate(env_state)
            # М'яке обмеження компонентів перед підсумовуванням
            component_reward = np.clip(component_reward, -15.0, 15.0)
            component_rewards[scheme.__class__.__name__] = component_reward
            total_reward += component_reward
        
        # КРИТИЧНО ВАЖЛИВО: Контроль від'ємної продуктивності
        total_return = env_state.get('total_return', 0.0)
        total_trades = env_state.get('total_trades', 0)
        
        # Лічильник для summary логування
        if not hasattr(self, 'negative_override_count'):
            self.negative_override_count = 0
            self.last_summary_step = 0
        
        # ВИМКНЕНО: Негативний override для стабільності навчання
        # Замість жорстких штрафів використовуємо м'які сигнали через основні компоненти
        # if total_return < -0.20 and total_trades > 10:  # Тільки при катастрофічних втратах >20%
        #     negative_override = total_return * 2.0  # М'який штраф
        #     negative_override = max(negative_override, -1.0)  # Максимум -1.0
        #     
        #     # Застосовуємо override тільки якщо початкова винагорода була позитивною
        #     if total_reward > negative_override:
        #         total_reward = negative_override
        #         self.negative_override_count += 1
        #         
        #         # ТІЛЬКИ summary логування кожні 50 overrides
        #         current_step = env_state.get('step', 0)
        #         if (self.negative_override_count % 50 == 0 or 
        #             current_step - self.last_summary_step > 1000):
        #             print(f"📊 NEGATIVE PERFORMANCE SUMMARY:")
        #             print(f"   Overrides applied: {self.negative_override_count}")
        #             print(f"   Current return: {total_return:.2%}, trades: {total_trades}")
        #             print(f"   Latest override: {negative_override:.2f}")
        #             self.last_summary_step = current_step
        
        # ДОДАТКОВІ АГРЕСИВНІ ШТРАФИ за погану продуктивність
        current_drawdown = env_state.get('max_drawdown', 0.0)
        win_rate = env_state.get('win_rate', 0.0)
        
        # ВИМКНЕНО: Всі додаткові штрафи для стабільності навчання
        # Основні компоненти схеми винагород вже включають контроль ризиків
        # Додаткові штрафи створюють нестабільність та заважають навчанню
        
        # # М'який штраф за критичну просадку (тільки при >30%)
        # if current_drawdown > 0.30:  
        #     drawdown_penalty = -current_drawdown * 5.0  # М'який штраф
        #     total_reward += drawdown_penalty
        
        # # М'який штраф за катастрофічні результати
        # if total_return < -0.50 and total_trades > 20 and win_rate < 0.2:  
        #     winrate_penalty = -(0.2 - win_rate) * 5.0  # Дуже м'який штраф
        #     total_reward += winrate_penalty
        
        # 🎯 СИСТЕМА ПРОГРЕСИВНИХ НАГОРОД ПОВНІСТЮ ВИМКНЕНА
        # improvement_bonus = self._calculate_improvement_bonus(env_state)
        # total_reward += improvement_bonus
        
        # ВИМКНЕНО: Система ескалації штрафів (створює нестабільність)
        # escalation_penalty = self._calculate_escalation_penalty(env_state)
        # total_reward += escalation_penalty
        
        # ВИМКНЕНО: Динамічне масштабування (створює нестабільність)
        # Використовуємо просте м'яке обмеження для стабільності
        scaled_reward = np.tanh(total_reward / 5.0) * 2.0  # М'яке обмеження до ±2.0
        
        # Мінімальний шум для варіативності
        noise = np.random.normal(0, 0.005)
        final_reward = scaled_reward + noise
        
        self.reward_history.append(final_reward)
        return final_reward
    
    def _calculate_improvement_bonus(self, env_state: Dict) -> float:
        """ВИПРАВЛЕНА логіка бонусів - ВИМКНЕНО міжепізодні порівняння."""
        # ПОВНІСТЮ ВИМКНЕНО систему покращень через некоректні порівняння
        # Система порівнювала результати різних епізодів, створюючи ложні "покращення"
        # Наприклад: -50% в одному епізоді → +20% в наступному = "покращення" на 70%
        # Це призводило до розбіжностей між показниками total_return та PROFIT IMPROVEMENT
        
        return 0.0  # ВИМКНЕНО всі бонуси за покращення
    
    def _calculate_escalation_penalty(self, env_state: Dict) -> float:
        """Розраховує ескалаційний штраф за послідовну погану продуктивність."""
        current_return = env_state.get('total_return', 0.0)
        current_step = env_state.get('step', 0)
        
        # Перевіряємо чи поточна продуктивність погана
        if current_return < self.poor_performance_threshold:
            self.consecutive_poor_episodes += 1
        else:
            # Скидаємо лічильник якщо продуктивність покращилася
            self.consecutive_poor_episodes = 0
            return 0.0
        
        # Розраховуємо ескалаційний штраф
        escalation_penalty = 0.0
        
        if self.consecutive_poor_episodes >= 10:  # 10+ поганих епізодів підряд
            # ЖОРСТКИЙ ШТРАФ за тривалу погану продуктивність
            escalation_multiplier = min(self.consecutive_poor_episodes / 10.0, 5.0)  # До 5x множника
            base_penalty = abs(current_return) * 30.0  # Базовий штраф
            escalation_penalty = -base_penalty * escalation_multiplier
            escalation_penalty = max(escalation_penalty, -50.0)  # Максимум -50.0 штрафу
            
            # Логування кожні 50 поганих епізодів
            if self.consecutive_poor_episodes % 50 == 0:
                print(f"🔥 ESCALATION PENALTY: {self.consecutive_poor_episodes} consecutive poor episodes")
                print(f"   Return: {current_return:.2%}, Penalty: {escalation_penalty:.2f}")
        
        elif self.consecutive_poor_episodes >= 5:  # 5-9 поганих епізодів
            # Помірний додатковий штраф
            escalation_penalty = -abs(current_return) * 10.0
            escalation_penalty = max(escalation_penalty, -10.0)
        
        return escalation_penalty
    
    def reset(self):
        """Скидання історії винагород."""
        self.reward_history = []
        if self.enable_dynamic_scaling:
            self.scaler.reset()
        for scheme in self.schemes:
            scheme.reset()
        
        # ВИПРАВЛЕНО: Скидаємо систему прогресивних нагород до РЕАЛЬНИХ початкових значень
        self.last_performance = {
            'return': 0.0,  # ВИПРАВЛЕНО: Починаємо з нульової доходності (не -100%)
            'drawdown': 0.0,  # ВИПРАВЛЕНО: Починаємо без просадки
            'trades': 0,
            'win_rate': 0.0
        }
        self.improvement_bonus_count = 0
        
        # Скидаємо систему ескалації штрафів
        self.consecutive_poor_episodes = 0


class SharpeRatioReward(BaseRewardScheme):
    """Винагорода на основі коефіцієнта Шарпа."""
    
    def __init__(self, weight: float = 0.3, window: int = 50, risk_free_rate: float = 0.02):
        super().__init__(weight)
        self.window = window
        self.risk_free_rate = risk_free_rate / 252  # Денна безризикова ставка
        self.returns_history = []
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < 2:
            return 0.0
        
        # Розраховуємо денні доходності
        returns = np.diff(portfolio_history) / portfolio_history[:-1]
        self.returns_history.extend(returns[-1:])  # Додаємо тільки останню доходність
        
        # Обмежуємо розмір історії
        if len(self.returns_history) > self.window:
            self.returns_history = self.returns_history[-self.window:]
        
        if len(self.returns_history) < 10:  # Мінімум для розрахунку
            return 0.0
        
        # Розраховуємо коефіцієнт Шарпа
        excess_returns = np.array(self.returns_history) - self.risk_free_rate
        if np.std(excess_returns) > 0:
            sharpe = np.mean(excess_returns) / np.std(excess_returns)
            return np.tanh(sharpe) * self.weight
        
        return 0.0
    
    def reset(self):
        self.returns_history = []


class TradeQualityReward(BaseRewardScheme):
    """Винагорода за якість угод."""
    
    def __init__(self, weight: float = 0.2, min_trades: int = 5):
        super().__init__(weight)
        self.min_trades = min_trades
    
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 0.0)
        
        if total_trades < self.min_trades:
            return 0.0
        
        # Бонус за високу долю прибуткових угод
        if win_rate > 0.6:
            return (win_rate - 0.5) * 2 * self.weight
        elif win_rate < 0.4:
            return -(0.5 - win_rate) * 2 * self.weight
        
        return 0.0


class VolatilityPenalty(BaseRewardScheme):
    """Штраф за високу волатильність портфеля."""
    
    def __init__(self, weight: float = -0.1, window: int = 20):
        super().__init__(weight)
        self.window = window
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < self.window:
            return 0.0
        
        # Розраховуємо волатільність останніх значень
        recent_values = portfolio_history[-self.window:]
        returns = np.diff(recent_values) / recent_values[:-1]
        volatility = np.std(returns)
        
        # Штраф за високу волатільність
        if volatility > 0.05:  # 5% денна волатільність
            return -volatility * 10 * abs(self.weight)
        
        return 0.0


class ConsistencyReward(BaseRewardScheme):
    """Винагорода за консистентність прибутку."""
    
    def __init__(self, weight: float = 0.15, window: int = 30):
        super().__init__(weight)
        self.window = window
    
    def calculate(self, env_state: Dict) -> float:
        portfolio_history = env_state.get('portfolio_history', [])
        
        if len(portfolio_history) < self.window:
            return 0.0
        
        # Аналізуємо останні значення
        recent_values = portfolio_history[-self.window:]
        returns = np.diff(recent_values) / recent_values[:-1]
        
        # Доля позитивних днів
        positive_days_ratio = np.sum(returns > 0) / len(returns)
        
        # Бонус за консистентність
        if positive_days_ratio > 0.6:
            return (positive_days_ratio - 0.5) * 2 * self.weight
        
        return 0.0


class PerformanceDeclineReward(BaseRewardScheme):
    """
    Динамічна винагорода що зменшується при погіршенні ключових метрик:
    - Кількість угод
    - Win rate (відсоток прибуткових угод)
    - Просадка
    """
    
    def __init__(self, weight: float = -2.0, history_window: int = 10, decline_threshold: float = 0.15):
        super().__init__(weight)
        self.history_window = history_window
        self.decline_threshold = decline_threshold  # 15% погіршення
        
        # Історія метрик
        self.trades_history = []
        self.win_rate_history = []
        self.drawdown_history = []
        
        self.last_trades = 0
        self.last_win_rate = 0.0
        self.last_max_drawdown = 0.0
    
    def calculate(self, env_state: Dict) -> float:
        current_trades = env_state.get('total_trades', 0)
        current_win_rate = env_state.get('win_rate', 0.0)
        current_max_drawdown = env_state.get('max_drawdown', 0.0)
        
        # Оновлюємо історію метрик
        self.trades_history.append(current_trades)
        self.win_rate_history.append(current_win_rate)
        self.drawdown_history.append(current_max_drawdown)
        
        # Обмежуємо розмір історії
        if len(self.trades_history) > self.history_window:
            self.trades_history = self.trades_history[-self.history_window:]
            self.win_rate_history = self.win_rate_history[-self.history_window:]
            self.drawdown_history = self.drawdown_history[-self.history_window:]
        
        # Недостатньо історії для аналізу
        if len(self.trades_history) < 3:
            return 0.0
        
        total_penalty = 0.0
        
        # 1. АНАЛІЗ КІЛЬКОСТІ УГОД (перевіряємо зменшення активності)
        if len(self.trades_history) >= 6:
            recent_trades_data = self.trades_history[-3:]
            older_trades_data = self.trades_history[-6:-3]
            
            if len(recent_trades_data) >= 3 and len(older_trades_data) >= 3:
                recent_trades = np.mean(recent_trades_data)
                older_trades = np.mean(older_trades_data)
                
                if older_trades > 0:
                    trades_change = (recent_trades - older_trades) / older_trades
                    if trades_change < -self.decline_threshold:  # Зменшення активності на 15%+
                        penalty = abs(trades_change) * 2.0  # Штраф пропорційний зменшенню
                        total_penalty += penalty
                        print(f"🔻 TRADE ACTIVITY DECLINE: {trades_change:.1%} -> Penalty: {penalty:.2f}")
        
        # 2. АНАЛІЗ WIN RATE (перевіряємо зменшення ефективності)
        if len(self.win_rate_history) >= 6:
            recent_win_rate_data = [x for x in self.win_rate_history[-3:] if x > 0]
            older_win_rate_data = [x for x in self.win_rate_history[-6:-3] if x > 0]
            
            # ВИПРАВЛЕННЯ: Перевіряємо що списки не пусті перед np.mean
            if len(recent_win_rate_data) > 0 and len(older_win_rate_data) > 0:
                recent_win_rate = np.mean(recent_win_rate_data)
                older_win_rate = np.mean(older_win_rate_data)
                
                if older_win_rate > 0.1 and recent_win_rate > 0:  # Тільки якщо є значущі дані
                    win_rate_change = (recent_win_rate - older_win_rate) / older_win_rate
                    if win_rate_change < -self.decline_threshold:  # Зменшення винрейту на 15%+
                        penalty = abs(win_rate_change) * 3.0  # Більший штраф за погіршення якості
                        total_penalty += penalty
                        print(f"🔻 WIN RATE DECLINE: {win_rate_change:.1%} -> Penalty: {penalty:.2f}")
        
        # 3. АНАЛІЗ ПРОСАДКИ (перевіряємо збільшення ризику)
        if len(self.drawdown_history) >= 6:
            recent_drawdown_data = self.drawdown_history[-3:]
            older_drawdown_data = self.drawdown_history[-6:-3]
            
            if len(recent_drawdown_data) >= 3 and len(older_drawdown_data) >= 3:
                recent_drawdown = np.mean(recent_drawdown_data)
                older_drawdown = np.mean(older_drawdown_data)
                
                if older_drawdown > 0.001:  # Тільки якщо була просадка
                    drawdown_change = (recent_drawdown - older_drawdown) / older_drawdown
                    if drawdown_change > self.decline_threshold:  # Збільшення просадки на 15%+
                        penalty = drawdown_change * 2.5  # Штраф за збільшення ризику
                        total_penalty += penalty
                        print(f"🔻 DRAWDOWN INCREASE: {drawdown_change:.1%} -> Penalty: {penalty:.2f}")
        
        # Повертаємо штраф (негативну винагороду)
        final_penalty = -total_penalty * abs(self.weight) if total_penalty > 0 else 0.0
        
        # Логування значних штрафів
        if final_penalty < -1.0:
            print(f"📉 PERFORMANCE DECLINE PENALTY: {final_penalty:.2f}")
        
        return final_penalty
    
    def reset(self):
        self.trades_history = []
        self.win_rate_history = []
        self.drawdown_history = []
        self.last_trades = 0
        self.last_win_rate = 0.0
        self.last_max_drawdown = 0.0


def create_default_reward_scheme() -> CompositeRewardScheme:
    """Створити стандартну схему винагород."""
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
    """Створити консервативну схему винагород (акцент на стабільність)."""
    schemes = [
        ProfitReward(weight=0.7),
        DrawdownPenalty(weight=-1.0),
        SharpeRatioReward(weight=0.5),
        VolatilityPenalty(weight=-0.3),
        ConsistencyReward(weight=0.4)
    ]
    return CompositeRewardScheme(schemes)


def create_aggressive_reward_scheme() -> CompositeRewardScheme:
    """Створити агресивну схему винагород (акцент на прибуток)."""
    schemes = [
        ProfitReward(weight=1.5),
        DrawdownPenalty(weight=-0.2),
        TradeQualityReward(weight=0.3),
        SharpeRatioReward(weight=0.2)
    ]
    return CompositeRewardScheme(schemes)


class StaticReward(BaseRewardScheme):
    """Проста статична схема винагород з чіткими позитивними/негативними винагородами."""
    
    def __init__(self, weight: float = 1.0, static_initial_balance: float = None):
        super().__init__(weight)
        self.step_count = 0
        self.last_portfolio_value = None
        self.static_initial_balance = static_initial_balance  # ВИПРАВЛЕННЯ: статичний початковий баланс
        self.initial_balance = None
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state['portfolio_value']
        current_step = env_state.get('step', 0)
        
        # ВИПРАВЛЕННЯ: Використовуємо СТАТИЧНИЙ початковий баланс
        if self.initial_balance is None:
            # Використовуємо статичний баланс або беремо з env_state
            if self.static_initial_balance is not None:
                self.initial_balance = self.static_initial_balance
            else:
                # Отримуємо початковий баланс з конфігурації середовища
                self.initial_balance = env_state.get('initial_balance', 10000.0)
            self.last_portfolio_value = portfolio_value
            
        self.step_count += 1
        
        # ВИПРАВЛЕННЯ: Безпечний розрахунок зміни вартості портфеля
        step_change = portfolio_value - self.last_portfolio_value if self.last_portfolio_value is not None else 0
        step_change_percent = step_change / self.last_portfolio_value if self.last_portfolio_value is not None and self.last_portfolio_value > 0 else 0
        
        # Розрахуємо загальну прибутковість
        total_return_percent = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # ДУЖЕ ПРОСТА статична винагорода з гарантованою варіацією
        base_reward = 0.0
        
        # 1. Основна винагорода залежить від зміни портфеля
        if abs(step_change_percent) > 0.005:  # Значна зміна > 0.5%
            if step_change_percent > 0:
                base_reward = 3.0  # Великий прибуток
            else:
                base_reward = -3.0  # Великий збиток
        elif abs(step_change_percent) > 0.001:  # Помірна зміна > 0.1%
            if step_change_percent > 0:
                base_reward = 1.0  # Помірний прибуток
            else:
                base_reward = -1.0  # Помірний збиток
        elif abs(step_change_percent) > 0.0001:  # Мала зміна > 0.01%
            if step_change_percent > 0:
                base_reward = 0.5  # Малий прибуток
            else:
                base_reward = -0.5  # Малий збиток
        else:
            base_reward = -0.2  # Штраф за відсутність активності
        
        # 2. Додаємо компонент, що змінюється від кроку до кроку
        step_variation = np.sin(current_step * 0.1) * 0.3  # Варіація ±0.3
        
        # 3. Додаємо компонент залежний від загальної прибутковості
        performance_bonus = 0.0
        if total_return_percent > 0.1:  # > 10% прибуток
            performance_bonus = 2.0
        elif total_return_percent > 0.05:  # > 5% прибуток
            performance_bonus = 1.0
        elif total_return_percent < -0.1:  # > 10% збиток
            performance_bonus = -2.0
        elif total_return_percent < -0.05:  # > 5% збиток
            performance_bonus = -1.0
        
        # 4. Додаємо невеликий рандомний компонент для гарантованої варіації
        random_component = np.random.uniform(-0.2, 0.2)
        
        # Підсумкова винагорода
        final_reward = base_reward + step_variation + performance_bonus + random_component
        
        # Мінімальне логування - тільки при значних змінах або періодично
        if current_step % 1000 == 0 and current_step > 0:  # Виводимо тільки кожні 1000 кроків
            print(f"Step {current_step}: Portfolio: {portfolio_value:.2f}, Return: {total_return_percent:.2%}, Reward: {final_reward:.2f}")
        
        self.last_portfolio_value = portfolio_value
        
        # Обмежуємо винагороду для стабільності навчання
        final_reward = np.clip(final_reward * self.weight, -10.0, 10.0)
        return final_reward
    
    def reset(self):
        self.step_count = 0
        self.last_portfolio_value = None
        # ВИПРАВЛЕННЯ: НЕ скидаємо initial_balance, щоб зберегти статичність
        # self.initial_balance = None  # Закоментовано для збереження статичного балансу


class AdaptiveTradeOffReward(BaseRewardScheme):
    """
    Адаптивна винагорода що дозволяє моделі вибирати між різними цілями:
    - Прибуток vs Просадка
    - Активність vs Якість угод
    - Стабільність vs Агресивність
    """
    
    def __init__(self, weight: float = 10.0, adaptation_window: int = 50):
        super().__init__(weight)
        self.adaptation_window = adaptation_window
        self.performance_history = []
        self.current_strategy = "balanced"  # balanced, profit_focused, risk_focused
        self.strategy_counter = 0
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state.get('portfolio_value', 10000)
        total_return = env_state.get('total_return', 0.0)
        max_drawdown = env_state.get('max_drawdown', 0.0)
        win_rate = env_state.get('win_rate', 0.0)
        total_trades = env_state.get('total_trades', 0)
        
        # ВИПРАВЛЕННЯ: Фільтруємо безглузді записи з нульовими значеннями
        # Зберігаємо тільки значущі дані (з торговою активністю або значущими змінами)
        is_meaningful_data = (
            total_trades > 0 or  # Є торгова активність
            abs(total_return) > 0.001 or  # Є значущі зміни доходності (>0.1%)
            max_drawdown > 0.001  # Є значуща просадка (>0.1%)
        )
        
        if is_meaningful_data:
            self.performance_history.append({
                'return': total_return,
                'drawdown': max_drawdown,
                'win_rate': win_rate,
                'trades': total_trades
            })
            
            # Обмежуємо історію
            if len(self.performance_history) > self.adaptation_window:
                self.performance_history = self.performance_history[-self.adaptation_window:]
        
        # Адаптивний вибір стратегії кожні 20 кроків (тільки якщо є достатньо значущих даних)
        self.strategy_counter += 1
        if self.strategy_counter % 20 == 0 and len(self.performance_history) > 5:
            self.current_strategy = self._choose_strategy()
        
        # Розраховуємо винагороду згідно поточної стратегії
        reward = 0.0
        
        if self.current_strategy == "profit_focused":
            # Акцент на прибуток
            reward = total_return * 15.0  # Сильна винагорода за прибуток
            if max_drawdown > 0.15:  # Тільки критична просадка карається
                reward -= max_drawdown * 5.0
                
        elif self.current_strategy == "risk_focused":
            # Акцент на контроль ризиків
            reward = total_return * 8.0  # Помірна винагорода за прибуток
            reward -= max_drawdown * 20.0  # Сильне покарання за просадку
            if win_rate > 0.7:  # Бонус за високу якість
                reward += win_rate * 3.0
                
        else:  # balanced
            # Збалансований підхід з trade-off між цілями
            profit_component = total_return * 12.0
            risk_component = -max_drawdown * 10.0
            quality_component = win_rate * 2.0 if total_trades > 0 else 0
            
            # Дозволяємо моделі вибрати пріоритет
            if total_return > 0.05:  # При хорошому прибутку - дозволяємо більшу просадку
                risk_component *= 0.5
            elif max_drawdown < 0.03:  # При низькій просадці - заохочуємо більший ризик
                profit_component *= 1.5
                
            reward = profit_component + risk_component + quality_component
        
        return reward * self.weight
    
    def _choose_strategy(self) -> str:
        """Вибирає найкращу стратегію на основі недавньої продуктивності."""
        if len(self.performance_history) < 10:
            return "balanced"
        
        recent_performance = self.performance_history[-10:]
        
        # ВИПРАВЛЕННЯ: Безпечний розрахунок середніх значень з перевіркою на пусті масиви
        returns = [p['return'] for p in recent_performance]
        drawdowns = [p['drawdown'] for p in recent_performance]
        win_rates = [p['win_rate'] for p in recent_performance if p['win_rate'] > 0]
        
        # Розраховуємо середні з перевіркою на пусті списки
        avg_return = np.mean(returns) if len(returns) > 0 else 0.0
        avg_drawdown = np.mean(drawdowns) if len(drawdowns) > 0 else 0.0
        avg_win_rate = np.mean(win_rates) if len(win_rates) > 0 else 0.0
        
        # Логіка вибору стратегії
        if avg_return < -0.02 and avg_drawdown > 0.1:
            # Погана продуктивність - фокус на ризики
            print(f"🛡️ STRATEGY: Switching to RISK_FOCUSED (return={avg_return:.2%}, dd={avg_drawdown:.2%})")
            return "risk_focused"
        elif avg_return > 0.05 and avg_drawdown < 0.08:
            # Хороша продуктивність - фокус на прибуток
            print(f"💰 STRATEGY: Switching to PROFIT_FOCUSED (return={avg_return:.2%}, dd={avg_drawdown:.2%})")
            return "profit_focused"
        else:
            # Збалансований підхід
            print(f"⚖️ STRATEGY: Staying BALANCED (return={avg_return:.2%}, dd={avg_drawdown:.2%})")
            return "balanced"
    
    def reset(self):
        self.performance_history = []
        self.current_strategy = "balanced"
        self.strategy_counter = 0


def create_optimized_reward_scheme() -> CompositeRewardScheme:
    """
    ПОКРАЩЕНА схема винагород для максимізації прибутковості:
    - Фокус на прибуток з мінімальними штрафами
    - Зменшені penalty компоненти
    - Збільшена толерантність до ризиків
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Прибуток з підвищеною вагою
        ProfitReward(weight=2.0, normalize=True),  # Збільшено вагу прибутку
        
        # М'який контроль просадки тільки при критичних рівнях
        DrawdownPenalty(weight=-0.3, max_drawdown_threshold=0.25),  # Збільшено поріг до 25%
        
        # Винагорода за якість угод з м'якими умовами
        TradeQualityReward(weight=0.3, min_trades=1),  # Зменшено мінімум угод
        
        # Мінімальний Sharpe bonus
        SharpeRatioReward(weight=0.2, window=30),  # Зменшена вага
        
        # ВИМКНЕНО агресивні penalty компоненти:
        # - PerformanceDeclineReward (занадто штрафує за природні коливання)
        # - VolatilityPenalty (обмежує торгову активність)
        # - ConsistencyReward (може штрафувати за агресивну торгівлю)
    ]
    
    # ВИМКНЕНО динамічне масштабування для більшої стабільності
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=False)
    
    return composite


def create_bear_market_optimized_reward_scheme() -> CompositeRewardScheme:
    """
    СПЕЦІАЛЬНА схема винагород для падаючих ринків - ВИПРАВЛЕНА:
    - Максимальне заохочення ТІЛЬКИ за реальний прибуток портфеля
    - Мінімальні штрафи за ризики для збереження активності
    - Бонуси за прибуткову торгівлю в складних ринкових умовах
    - ВИМКНЕНО винагороди за саме падіння цін
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Агресивна винагорода за прибуток портфеля
        ProfitReward(weight=3.0, normalize=True),  # Потрійна вага - основний драйвер навчання
        
        # МІНІМАЛЬНИЙ контроль просадки тільки при критичних рівнях
        DrawdownPenalty(weight=-0.1, max_drawdown_threshold=0.35),  # Дуже високий поріг 35% - не заважає активній торгівлі
        
        # ВИНАГОРОДА за якість угод (високий винрейт)
        TradeQualityReward(weight=0.5, min_trades=1),  # Заохочуємо якісні рішення
        
        # ВИПРАВЛЕНИЙ БОНУС за прибуткову торгівлю при падаючому ринку
        BearMarketActivityReward(weight=1.0),  # Тепер винагороджує ТІЛЬКИ за прибуток, а не за падіння
        
        # ВИПРАВЛЕНИЙ БОНУС за правильний тайминг та прибуткові операції
        MarketTimingReward(weight=0.8),  # Тепер базується на прибутковості портфеля
    ]
    
    # Без динамічного масштабування для максимальної стабільності навчання
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=False)
    
    return composite


class BearMarketActivityReward(BaseRewardScheme):
    """Винагорода за прибуткову торгівлю під час падаючого ринку - ВИПРАВЛЕНА ЛОГІКА."""
    
    def __init__(self, weight: float = 1.0, market_decline_threshold: float = -0.05):
        super().__init__(weight)
        self.market_decline_threshold = market_decline_threshold  # -5% спад за період для визначення ведмежого ринку
        self.price_history = []
        self.last_portfolio_value = None
        
    def calculate(self, env_state: Dict) -> float:
        current_price = env_state.get('current_price', 0)
        total_trades = env_state.get('total_trades', 0)
        portfolio_value = env_state.get('portfolio_value', 0)
        
        if current_price <= 0:
            return 0.0
            
        # Ведемо історію цін для визначення ринкового тренду
        self.price_history.append(current_price)
        if len(self.price_history) > 20:  # Останні 20 кроків
            self.price_history = self.price_history[-20:]
            
        if len(self.price_history) < 10:
            # Ініціалізуємо початкове значення портфеля
            if self.last_portfolio_value is None:
                self.last_portfolio_value = portfolio_value
            return 0.0
            
        # Визначаємо чи ринок падає (ведмежий тренд)
        price_change = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        is_bear_market = price_change < self.market_decline_threshold
        
        # Розраховуємо зміну портфеля з попереднього кроку
        portfolio_change = 0.0
        if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
            portfolio_change = portfolio_value - self.last_portfolio_value
        
        reward = 0.0
        
        if is_bear_market:
            # ВИПРАВЛЕНО: Винагорода ТІЛЬКИ за реальний ПРИБУТОК портфеля при падаючому ринку
            if portfolio_change > 0:
                # Великий бонус за збільшення портфеля під час падіння ринку
                profit_percentage = portfolio_change / self.last_portfolio_value
                profit_bonus = profit_percentage * 50.0  # 1% прибутку = +0.5 бонусу
                reward += min(profit_bonus, 8.0)  # Максимум +8.0
                
                # Додатковий бонус за торгову активність при прибутковій торгівлі
                if total_trades > 0:
                    activity_bonus = min(total_trades * 0.1, 1.5)  # До +1.5 за активність
                    reward += activity_bonus
                    
            # ВИПРАВЛЕНО: Малий штраф за втрати при падаючому ринку (м'який спосіб навчити уникати втрат)
            elif portfolio_change < -100:  # Тільки при значних втратах > $100
                loss_percentage = abs(portfolio_change) / self.last_portfolio_value
                loss_penalty = -loss_percentage * 5.0  # М'який штраф за втрати
                reward += max(loss_penalty, -2.0)  # Максимум -2.0 штрафу
            
            # ВИМКНЕНО: НЕПРАВИЛЬНА логіка винагород за падіння цін
            # НЕ ДОДАЄМО бонуси за само падіння ринку!
            # Агент має отримувати винагороди тільки за прибуток портфеля!
            
        # Оновлюємо історію портфеля
        self.last_portfolio_value = portfolio_value
        
        return reward * self.weight
        
    def reset(self):
        self.price_history = []


class MarketTimingReward(BaseRewardScheme):
    """Винагорода за правильний тайминг операцій - ВИПРАВЛЕНА ЛОГІКА."""
    
    def __init__(self, weight: float = 0.8):
        super().__init__(weight)
        self.last_action = None
        self.last_price = None
        self.last_portfolio_value = None
        
    def calculate(self, env_state: Dict) -> float:
        current_price = env_state.get('current_price', 0)
        current_action = env_state.get('last_action', 0)  # 0=утримувати, 1=купувати, 2=продавати
        portfolio_value = env_state.get('portfolio_value', 10000)
        
        # Ініціалізація при першому виклику
        if current_price <= 0 or self.last_price is None:
            self.last_price = current_price
            self.last_action = current_action
            self.last_portfolio_value = portfolio_value
            return 0.0
            
        # Розраховуємо зміну ціни та портфеля
        price_change = (current_price - self.last_price) / self.last_price
        portfolio_change = 0.0
        if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
            portfolio_change = portfolio_value - self.last_portfolio_value
        
        reward = 0.0
        
        # ВИПРАВЛЕНО: Винагорода ТІЛЬКИ за реальне збільшення портфеля
        if portfolio_change > 0:
            # Розраховуємо відсоток прибутку
            profit_percentage = portfolio_change / self.last_portfolio_value
            
            # Базовий бонус за прибуток
            base_profit_bonus = profit_percentage * 20.0  # 1% прибутку = +0.2 бонусу
            reward += min(base_profit_bonus, 5.0)  # Максимум +5.0
            
            # ДОДАТКОВИЙ бонус за прибуток при падаючому ринку (складніша умова)
            if price_change < -0.02:  # Ціна впала більше ніж на 2%
                bear_market_bonus = profit_percentage * 30.0  # Додатковий множник за торгівлю проти тренду
                reward += min(bear_market_bonus, 3.0)  # Максимум +3.0 додатково
                
            # БОНУС за правильний тайминг угод (якщо дія відповідає ринку)
            if self.last_action == 1 and price_change > 0.01:  # Купували перед зростанням
                timing_bonus = profit_percentage * 15.0
                reward += min(timing_bonus, 2.0)  # Максимум +2.0
            elif self.last_action == 2 and price_change < -0.01:  # Продавали перед падінням
                timing_bonus = profit_percentage * 15.0
                reward += min(timing_bonus, 2.0)  # Максимум +2.0
        
        # ВИПРАВЛЕНО: М'який штраф за втрати (тільки при значних втратах)
        elif portfolio_change < -50:  # Втрати більше $50
            loss_percentage = abs(portfolio_change) / self.last_portfolio_value
            loss_penalty = -loss_percentage * 10.0  # М'який штраф за втрати
            reward += max(loss_penalty, -3.0)  # Максимум -3.0 штрафу
        
        # ВИМКНЕНО: Неправильна логіка винагород за падіння цін
        # Агент НЕ має отримувати винагороди за саме падіння ринку!
        # Винагороди мають базуватися ТІЛЬКИ на прибутковості портфеля!
        
        # Оновлюємо стан для наступного кроку
        self.last_price = current_price
        self.last_action = current_action
        self.last_portfolio_value = portfolio_value
        
        return reward * self.weight
        
    def reset(self):
        """Скидання стану схеми при початку нового епізоду."""
        self.last_action = None
        self.last_price = None
        self.last_portfolio_value = None


def create_static_reward_scheme(initial_balance: float = 10000.0) -> StaticReward:
    """Створити просту статичну схему винагород з фіксованим початковим балансом."""
    return StaticReward(weight=1.0, static_initial_balance=initial_balance)