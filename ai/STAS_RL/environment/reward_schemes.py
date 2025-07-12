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
        """Розрахувати загальну винагороду з жорстким контролем збитків."""
        total_reward = 0.0
        component_rewards = {}
        
        for scheme in self.schemes:
            component_reward = scheme.calculate(env_state)
            # М'яке обмеження компонентів перед підсумовуванням
            component_reward = np.clip(component_reward, -15.0, 15.0)
            component_rewards[scheme.__class__.__name__] = component_reward
            total_reward += component_reward
        
        # КРИТИЧНО ВАЖЛИВИЙ КОНТРОЛЬ ЗБИТКІВ
        total_return = env_state.get('total_return', 0.0)
        total_trades = env_state.get('total_trades', 0)
        
        # 🎯 СТАБІЛЬНА АДАПТИВНА СИСТЕМА ВИНАГОРОД 🎯
        # Розумне масштабування для стабільного навчання з прогресивними винагородами
        
        # АДАПТИВНА базова винагорода з м'яким насиченням
        if abs(total_return) <= 0.05:  # Малі зміни: лінійне масштабування
            base_reward = total_return * 10.0  # 1% = 0.1 винагорода
        elif abs(total_return) <= 0.20:  # Помірні зміни: зменшене масштабування  
            sign = 1 if total_return > 0 else -1
            scaled_return = abs(total_return)
            base_reward = sign * (0.5 + (scaled_return - 0.05) * 6.67)  # Плавний перехід
        else:  # Великі зміни: логарифмічне насичення
            sign = 1 if total_return > 0 else -1
            scaled_return = abs(total_return)
            base_reward = sign * (1.5 + np.log(scaled_return * 5) * 0.8)  # М'яке насичення
        
        # СТАБІЛІЗУЮЧІ компоненти для зменшення варіативності
        win_rate = env_state.get('win_rate', 0.5)
        max_drawdown = env_state.get('max_drawdown', 0.0)
        
        # Бонус за стабільність (знижує варіативність)
        stability_bonus = 0.0
        if total_trades > 5:  # Тільки при достатній активності
            # Винагорода за збалансовану торгівлю
            if 0.4 <= win_rate <= 0.7 and max_drawdown < 0.15:
                stability_bonus = 0.3 * (1 - abs(win_rate - 0.55) * 4)  # Максимум при 55% винрейт
            
            # М'який штраф за крайнощі
            if win_rate < 0.3 or max_drawdown > 0.25:
                stability_bonus -= 0.2
        
        # ФІНАЛЬНА винагорода з контролем варіативності
        final_reward = base_reward + stability_bonus
        
        # ЗБІЛЬШЕНИЙ адаптивний шум для диференціації схожих результатів
        adaptive_noise = np.random.normal(0, 0.1)  # ЗБІЛЬШЕНО шум до ±0.1 для кращої варіативності
        final_reward += adaptive_noise
        
        # КАРДИНАЛЬНО РОЗШИРЕНІ межі для диференціації схожих результатів
        final_reward = np.clip(final_reward, -15.0, 25.0)  # ЗБІЛЬШЕНО діапазон для кращої диференціації стратегій
        
        self.reward_history.append(final_reward)
        return final_reward
                
        # Масштабування тільки для прибуткових результатів (>0.1%)
        if total_reward > 0:
            # Для позитивних винагород використовуємо м'яке масштабування
            scaled_reward = np.tanh(total_reward / 5.0) * 2.0
        else:
            # Для негативних винагород зберігаємо повну силу штрафу
            scaled_reward = max(total_reward, -3.0)  # Максимум -3.0 штрафу
        
        # Мінімальний шум
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


class TotalReturnReward(BaseRewardScheme):
    """Винагорода за загальну прибутковість портфеля."""
    
    def __init__(self, weight: float = 3.0, initial_balance: float = 10000.0):
        super().__init__(weight)
        self.initial_balance = initial_balance
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state.get('portfolio_value', self.initial_balance)
        
        # Отримуємо початковий баланс з середовища або використовуємо значення за замовчуванням
        initial_balance = env_state.get('initial_balance', self.initial_balance)
        
        # Розраховуємо загальну доходність
        total_return = (portfolio_value - initial_balance) / initial_balance
        
        # Прогресивна винагорода за прибуток з нелінійним зростанням
        if total_return > 0:
            # Позитивна винагорода за прибуток з бонусом за високі результати
            if total_return > 0.5:  # >50% прибуток
                reward = (total_return * 15.0 + 5.0) * self.weight  # Великий бонус
            elif total_return > 0.2:  # >20% прибуток
                reward = (total_return * 12.0 + 2.0) * self.weight  # Хороший бонус
            elif total_return > 0.1:  # >10% прибуток
                reward = (total_return * 10.0 + 1.0) * self.weight  # Помірний бонус
            else:  # 0-10% прибуток
                reward = total_return * 8.0 * self.weight  # Базова винагорода
        else:
            # М'який штраф за збитки (значно менший ніж винагорода за прибуток)
            reward = total_return * 3.0 * self.weight  # Менший штраф заохочує ризик
            
        return np.clip(reward, -5.0, 20.0)  # Обмежуємо діапазон
    
    def reset(self):
        pass


class StepProfitReward(BaseRewardScheme):
    """Винагорода за крокові покращення портфеля (збалансований підхід)."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.last_portfolio_value = None
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state.get('portfolio_value', 10000)
        
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
            
        # Крокова зміна портфеля
        step_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        # М'яка винагорода за крокові покращення
        if step_change > 0.005:  # >0.5% покращення
            reward = min(step_change * 20.0, 2.0) * self.weight  # Максимум +2.0
        elif step_change > 0:  # Маленькі покращення
            reward = step_change * 10.0 * self.weight
        elif step_change > -0.01:  # Маленькі втрати (-1%)
            reward = step_change * 2.0 * self.weight  # М'який штраф
        else:  # Великі втрати
            reward = step_change * 5.0 * self.weight  # Помірний штраф
            
        return np.clip(reward, -1.0, 2.0)  # Обмежуємо діапазон
    
    def reset(self):
        self.last_portfolio_value = None


class LossTradesPenalty(BaseRewardScheme):
    """АГРЕСИВНІ штрафи за збиткові угоди та загальні втрати."""
    
    def __init__(self, weight: float = -4.0):
        super().__init__(weight)
        
    def calculate(self, env_state: Dict) -> float:
        total_return = env_state.get('total_return', 0.0)
        total_trades = env_state.get('total_trades', 0)
        portfolio_value = env_state.get('portfolio_value', 10000)
        initial_balance = env_state.get('initial_balance', 10000)
        
        penalty = 0.0
        
        # 1. АГРЕСИВНІ штрафи за загальні збитки
        if total_return < 0:
            # Прогресивний штраф за збитки
            if total_return < -0.10:  # >10% збитків
                penalty += abs(total_return) * 15.0  # Критичний штраф
            elif total_return < -0.05:  # >5% збитків
                penalty += abs(total_return) * 10.0  # Високий штраф
            else:  # <5% збитків
                penalty += abs(total_return) * 5.0   # Помірний штраф
                
        # 2. ДОДАТКОВІ штрафи за втрати при активній торгівлі
        if total_trades > 5 and total_return < -0.02:  # >2% збитків при >5 угодах
            activity_penalty = abs(total_return) * total_trades * 0.1
            penalty += min(activity_penalty, 3.0)  # Максимум +3.0 штрафу
            
        # 3. КРИТИЧНІ штрафи за катастрофічні втрати
        if total_return < -0.15:  # >15% збитків
            catastrophic_penalty = abs(total_return) * 20.0
            penalty += min(catastrophic_penalty, 5.0)  # Максимум +5.0 штрафу
            
        # Повертаємо негативну винагороду (штраф)
        final_penalty = -penalty * abs(self.weight) if penalty > 0 else 0.0
        return np.clip(final_penalty, -15.0, 0.0)  # Обмежуємо діапазон штрафів
    
    def reset(self):
        pass


class WinRatePenalty(BaseRewardScheme):
    """ШТРАФИ за низький винрейт при активній торгівлі."""
    
    def __init__(self, weight: float = -2.0, min_trades: int = 5):
        super().__init__(weight)
        self.min_trades = min_trades
        
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 0.0)
        total_return = env_state.get('total_return', 0.0)
        
        # Тільки штрафуємо при достатній торговій активності
        if total_trades < self.min_trades:
            return 0.0
            
        penalty = 0.0
        
        # 1. ШТРАФИ за низький винрейт
        if win_rate < 0.4:  # <40% винрейт
            base_penalty = (0.4 - win_rate) * 5.0  # Штраф пропорційний відхиленню
            penalty += base_penalty
            
        # 2. ПОДВІЙНІ штрафи за низький винрейт при збитках
        if win_rate < 0.3 and total_return < 0:  # <30% винрейт + збитки
            double_penalty = (0.3 - win_rate) * 8.0  # Подвійний штраф
            penalty += double_penalty
            
        # 3. КРИТИЧНІ штрафи за катастрофічно низький винрейт
        if win_rate < 0.2 and total_trades > 10:  # <20% винрейт при активній торгівлі
            critical_penalty = (0.2 - win_rate) * 12.0  # Критичний штраф
            penalty += critical_penalty
            
        # Повертаємо негативну винагороду (штраф)
        final_penalty = -penalty * abs(self.weight) if penalty > 0 else 0.0
        return np.clip(final_penalty, -8.0, 0.0)  # Обмежуємо діапазон штрафів
    
    def reset(self):
        pass


class ExplorationReward(BaseRewardScheme):
    """Винагорода за торгову активність та експлорацію різних стратегій."""
    
    def __init__(self, weight: float = 0.5, target_trades_per_episode: int = 50):
        super().__init__(weight)
        self.target_trades_per_episode = target_trades_per_episode
        self.last_trades = 0
        
    def calculate(self, env_state: Dict) -> float:
        total_trades = env_state.get('total_trades', 0)
        current_step = env_state.get('step', 0)
        episode_length = env_state.get('ep_len_mean', 2000)
        
        # Розраховуємо прогрес торгової активності
        if episode_length > 0:
            expected_trades = (current_step / episode_length) * self.target_trades_per_episode
            trade_progress = total_trades / max(expected_trades, 1)
        else:
            trade_progress = 0
        
        reward = 0.0
        
        # 1. Винагорода за досягнення цільової активності
        if trade_progress >= 0.8:  # 80% від цільової активності
            reward += 1.0 * self.weight
        elif trade_progress >= 0.5:  # 50% від цільової активності
            reward += 0.5 * self.weight
        elif trade_progress >= 0.2:  # 20% від цільової активності
            reward += 0.2 * self.weight
        
        # 2. Бонус за нові угоди (заохочуємо активність)
        new_trades = total_trades - self.last_trades
        if new_trades > 0:
            activity_bonus = min(new_trades * 0.1, 0.5) * self.weight  # Максимум +0.5
            reward += activity_bonus
        
        # 3. М'який штраф за повну пасивність (тільки якщо немає угод взагалі)
        if total_trades == 0 and current_step > 500:  # Після 500 кроків без угод
            reward -= 0.2 * abs(self.weight)
        
        self.last_trades = total_trades
        return np.clip(reward, -0.5, 1.0)  # Обмежуємо діапазон
    
    def reset(self):
        self.last_trades = 0


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


class SimpleProfitReward(BaseRewardScheme):
    """Максимально проста схема винагород: прибуток = добре, збиток = погано."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.initial_balance = None
        
    def calculate(self, env_state: Dict) -> float:
        if self.initial_balance is None:
            self.initial_balance = env_state.get('initial_balance', 10000.0)
            
        portfolio_value = env_state.get('portfolio_value', self.initial_balance)
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # МАКСИМАЛЬНО ПРОСТА ЛОГІКА:
        # Прибуток = позитивна винагорода
        # Збиток = негативна винагорода
        # Більший прибуток = більша винагорода
        # Більший збиток = більший штраф
        
        if total_return > 0:
            # Прогресивна винагорода за прибуток
            if total_return > 0.20:      # >20% = відмінно
                reward = 10.0
            elif total_return > 0.10:    # >10% = дуже добре  
                reward = 5.0
            elif total_return > 0.05:    # >5% = добре
                reward = 2.0
            else:                        # >0% = добре
                reward = 1.0
        else:
            # Прогресивний штраф за збитки
            if total_return < -0.20:     # >20% збитків = катастрофа
                reward = -10.0
            elif total_return < -0.10:   # >10% збитків = дуже погано
                reward = -5.0
            elif total_return < -0.05:   # >5% збитків = погано
                reward = -2.0
            else:                        # <5% збитків = не страшно
                reward = -1.0
        
        # Логування для моніторингу
        step = env_state.get('step', 0)
        if step % 500 == 0:
            print(f"💡 SIMPLE PROFIT REWARD: return={total_return:+.1%} → reward={reward:.1f}")
            
        return reward * self.weight
    
    def reset(self):
        # НЕ скидаємо initial_balance для стабільності
        pass


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
    ПОКРАЩЕНА схема винагород для прогресивного навчання:
    - Основна винагорода за прибутковість з прогресивним масштабуванням
    - Бонуси за торгову активність та експлорацію
    - М'які штрафи за ризики для навчання балансу
    - Винагороди за покращення якості торгівлі
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Прогресивна винагорода за прибутковість
        TotalReturnReward(weight=4.0),  # Збільшена вага для сильних позитивних сигналів
        
        # ЕКСПЛОРАЦІЯ: Заохочення торгової активності
        ExplorationReward(weight=2.0, target_trades_per_episode=25),  # Активна торгівля
        
        # ЯКІСТЬ: Винагорода за хороші угоди
        TradeQualityReward(weight=1.5, min_trades=3),  # Бонус за високий винрейт
        
        # ПРОГРЕС: Винагорода за покращення між кроками
        StepProfitReward(weight=1.0),  # Заохочення позитивних змін
        
        # КОНТРОЛЬ РИЗИКІВ: М'який контроль просадки (не блокує навчання)
        DrawdownPenalty(weight=-1.0, max_drawdown_threshold=0.15),  # 15% поріг
        
        # БАЛАНС: М'які штрафи за великі втрати (заохочує обережність)
        LossTradesPenalty(weight=-1.5),  # Помірні штрафи за збитки
    ]
    
    # Увімкнути динамічне масштабування для стабільності навчання
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=True)
    return composite


def create_bear_market_optimized_reward_scheme() -> CompositeRewardScheme:
    """
    АГРЕСИВНО ОПТИМІЗОВАНА схема винагород для максимізації прибутку:
    - МАКСИМАЛЬНІ винагороди за прибуток портфеля  
    - АГРЕСИВНІ штрафи за збитки та втрати
    - ЖОРСТКИЙ контроль просадки
    - СУВОРІ покарання за погану торгівлю
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Агресивна винагорода за прибуток портфеля
        ProfitReward(weight=4.0, normalize=True),  # Збільшено до 4x для максимізації прибутку
        
        # АГРЕСИВНИЙ контроль просадки з низьким порогом
        DrawdownPenalty(weight=-2.0, max_drawdown_threshold=0.15),  # ЗНИЖЕНО поріг до 15%, збільшено штраф
        
        # ВИНАГОРОДА за якість угод (високий винрейт) 
        TradeQualityReward(weight=1.0, min_trades=1),  # Збільшено вагу якості
        
        # АГРЕСИВНІ штрафи за збиткову торгівлю
        AggressiveLossPenalty(weight=-3.0),  # НОВИЙ компонент для жорсткого покарання збитків
        
        # ВИПРАВЛЕНИЙ БОНУС за прибуткову торгівлю з АГРЕСИВНИМИ штрафами за втрати
        BearMarketActivityReward(weight=2.0),  # Збільшено вагу з агресивними штрафами
        
        # ВИПРАВЛЕНИЙ тайминг з ЖОРСТКИМИ штрафами за втрати
        MarketTimingReward(weight=1.5),  # Збільшено вагу з агресивними штрафами
    ]
    
    # Без динамічного масштабування для максимальної стабільності навчання
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=False)
    
    return composite


class AggressiveLossPenalty(BaseRewardScheme):
    """АГРЕСИВНІ штрафи за збиткові результати торгівлі."""
    
    def __init__(self, weight: float = -3.0, loss_threshold: float = -0.01):
        super().__init__(weight)
        self.loss_threshold = loss_threshold  # -1% як поріг для штрафів
        self.last_portfolio_value = None
        self.consecutive_losses = 0
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state.get('portfolio_value', 10000)
        total_return = env_state.get('total_return', 0.0)
        total_trades = env_state.get('total_trades', 0)
        win_rate = env_state.get('win_rate', 0.0)
        
        # Ініціалізація
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
            
        # Розраховуємо зміну портфеля з попереднього кроку
        step_change = 0.0
        if self.last_portfolio_value > 0:
            step_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            
        penalty = 0.0
        
        # 1. АГРЕСИВНІ штрафи за крокові втрати
        if step_change < self.loss_threshold:  # Втрати більше 1% за крок
            step_penalty = abs(step_change) * 50.0  # 1% втрат = -0.5 штрафу
            penalty += min(step_penalty, 5.0)  # Максимум -5.0 за крок
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # 2. ЕСКАЛАЦІЙНІ штрафи за послідовні втрати
        if self.consecutive_losses > 5:
            escalation = self.consecutive_losses * 0.2  # Збільшуємо штраф за кожну послідовну втрату
            penalty += min(escalation, 3.0)  # Максимум +3.0 до штрафу
            
        # 3. ЖОРСТКІ штрафи за загальну збитковість
        if total_return < -0.05 and total_trades > 10:  # Збитки більше 5% при активній торгівлі
            total_penalty = abs(total_return) * 30.0  # 1% збитків = -0.3 штрафу
            penalty += min(total_penalty, 8.0)  # Максимум -8.0
            
        # 4. ДОДАТКОВІ штрафи за низький винрейт при збитках
        if total_return < -0.02 and win_rate < 0.4 and total_trades > 5:
            winrate_penalty = (0.4 - win_rate) * 10.0  # Штраф за поганий винрейт
            penalty += min(winrate_penalty, 2.0)  # Максимум -2.0
            
        # Оновлюємо стан
        self.last_portfolio_value = portfolio_value
        
        # Повертаємо негативну винагороду (штраф)
        final_penalty = -penalty * abs(self.weight) if penalty > 0 else 0.0
        
        # Логування значних штрафів для моніторингу
        if final_penalty < -2.0:
            print(f"💥 AGGRESSIVE LOSS PENALTY: {final_penalty:.2f} (return: {total_return:.2%}, step_change: {step_change:.2%})")
            
        return final_penalty
        
    def reset(self):
        self.last_portfolio_value = None
        self.consecutive_losses = 0


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
            # АГРЕСИВНО ЗБІЛЬШЕНІ винагороди за прибуток портфеля
            if portfolio_change > 5:  # ЗНИЖЕНО поріг до $5 для заохочення
                # Винагорода пропорційна РЕАЛЬНОМУ прибутку
                profit_percentage = portfolio_change / self.last_portfolio_value
                # ЗБІЛЬШЕНО множники для максимізації прибутку
                profit_bonus = profit_percentage * 25.0  # 1% прибутку = +0.25 бонусу
                reward += min(profit_bonus, 5.0)  # ЗБІЛЬШЕНО максимум до +5.0
                    
            # КРИТИЧНО ЗБІЛЬШЕНІ штрафи за втрати портфеля  
            elif portfolio_change < -5:  # ЗНИЖЕНО поріг до $5 для жорсткості
                loss_percentage = abs(portfolio_change) / self.last_portfolio_value
                # АГРЕСИВНО ЗБІЛЬШЕНО штрафи за втрати
                loss_penalty = -loss_percentage * 60.0  # 1% втрат = -0.6 штрафу (було -0.15)
                reward += max(loss_penalty, -10.0)  # ЗБІЛЬШЕНО максимум штрафу до -10.0
            
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
        
        # АГРЕСИВНО ЗБІЛЬШЕНА винагорода за збільшення портфеля
        if portfolio_change > 3:  # ЗНИЖЕНО поріг до $3 для заохочення
            # Розраховуємо відсоток прибутку
            profit_percentage = portfolio_change / self.last_portfolio_value
            
            # ЗБІЛЬШЕНО базовий бонус за прибуток
            base_profit_bonus = profit_percentage * 20.0  # 1% прибутку = +0.2 бонусу (було +0.05)
            reward += min(base_profit_bonus, 4.0)  # ЗБІЛЬШЕНО максимум до +4.0
            
        # КРИТИЧНО ЗБІЛЬШЕНІ штрафи за втрати
        elif portfolio_change < -2:  # ЗНИЖЕНО поріг до $2 для жорсткості
            loss_percentage = abs(portfolio_change) / self.last_portfolio_value
            # АГРЕСИВНО ЗБІЛЬШЕНО штрафи за втрати
            loss_penalty = -loss_percentage * 80.0  # 1% втрат = -0.8 штрафу (було -0.2)
            reward += max(loss_penalty, -8.0)  # ЗБІЛЬШЕНО максимум штрафу до -8.0
        
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


def create_market_optimized_reward_scheme() -> CompositeRewardScheme:
    """
    ЗБАЛАНСОВАНА схема винагород для ефективного навчання:
    - Чіткі позитивні сигнали за прибуток
    - Помірні штрафи за збитки 
    - Заохочення експлорації та навчання
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Загальна прибутковість (відновлена вага)
        TotalReturnReward(weight=5.0),  # ЗБІЛЬШЕНО для сильних позитивних сигналів
        
        # ПОМІРНІ штрафи за збиткові результати
        LossTradesPenalty(weight=-2.0),  # ЗМЕНШЕНО для кращого навчання
        
        # М'ЯКІ ШТРАФИ за низький винрейт при активній торгівлі  
        WinRatePenalty(weight=-1.0),  # ЗМЕНШЕНО для заохочення експлорації
        
        # ТІЛЬКИ за критичну просадку
        DrawdownPenalty(weight=-0.5, max_drawdown_threshold=0.20),  # ЗБІЛЬШЕНО поріг до 20%
        
        # СИЛЬНА винагорода за якість угод
        TradeQualityReward(weight=2.0, min_trades=1),  # ЗБІЛЬШЕНО для заохочення
        
        # ПОЗИТИВНА вага для заохочення прибутковості
        StepProfitReward(weight=1.0),  # ПОВЕРНЕНО позитивну вагу
        
        # НОВИЙ: Винагорода за експлорацію (торгову активність)
        ExplorationReward(weight=0.5),  # Заохочуємо торгову активність
    ]
    
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=False)
    return composite


class CumulativeGrowthReward(BaseRewardScheme):
    """Винагорода за cumulative зростання портфеля через епізоди."""
    
    def __init__(self, weight: float = 4.0):
        super().__init__(weight)
        self.initial_balance = None
        
    def calculate(self, env_state: Dict) -> float:
        if self.initial_balance is None:
            self.initial_balance = env_state.get('initial_balance', 10000.0)
        
        # Використовуємо cumulative return якщо доступний, інакше episodic
        cumulative_return = env_state.get('cumulative_return', env_state.get('total_return', 0.0))
        portfolio_value = env_state.get('portfolio_value', 10000.0)
        
        # 💰 ЗБАЛАНСОВАНІ ВИНАГОРОДИ для стабільного навчання
        if cumulative_return > 0.2:  # >20% прибуток = ВЕЛИКА ВИНАГОРОДА
            reward = (cumulative_return * 10.0 + 3.0) * self.weight  # 20% = +5.0 винагорода
        elif cumulative_return > 0.1:  # >10% прибуток = ХОРОША ВИНАГОРОДА
            reward = (cumulative_return * 8.0 + 1.0) * self.weight   # 10% = +1.8 винагорода
        elif cumulative_return > 0.05:  # >5% прибуток = ПОМІРНА ВИНАГОРОДА
            reward = cumulative_return * 6.0 * self.weight           # 5% = +0.3 винагорода
        elif cumulative_return > 0:  # Будь-який прибуток = ПОЗИТИВНА ВИНАГОРОДА
            reward = cumulative_return * 4.0 * self.weight           # 1% = +0.04 винагорода
        else:  # 🔻 ПОМІРНІ ШТРАФИ ЗА ЗБИТКИ
            if cumulative_return < -0.2:  # >20% збитків = КРИТИЧНИЙ ШТРАФ
                reward = cumulative_return * 8.0 * self.weight       # -20% = -1.6 штраф
            elif cumulative_return < -0.1:  # >10% збитків = ВИСОКИЙ ШТРАФ
                reward = cumulative_return * 6.0 * self.weight       # -10% = -0.6 штраф
            else:  # <10% збитків = ПОМІРНИЙ ШТРАФ
                reward = cumulative_return * 4.0 * self.weight       # -5% = -0.2 штраф
            
        # Логування кожні 100 кроків
        step = env_state.get('step', 0)
        if step % 100 == 0:
            print(f"💰 CUMULATIVE GROWTH: return={cumulative_return:+.1%} → reward={reward:.2f} (portfolio=${portfolio_value:.0f})")
            
        return np.clip(reward, -10.0, 15.0)  # ЗБАЛАНСОВАНИЙ діапазон для стабільного навчання
    
    def reset(self):
        # НЕ скидаємо initial_balance для cumulative tracking
        pass


class CumulativeDrawdownPenalty(BaseRewardScheme):
    """Штраф за cumulative просадку портфеля."""
    
    def __init__(self, weight: float = -4.0, max_cumulative_drawdown: float = 0.20):
        super().__init__(weight)
        self.max_cumulative_drawdown = max_cumulative_drawdown
        
    def calculate(self, env_state: Dict) -> float:
        # Використовуємо cumulative drawdown якщо доступний
        cumulative_drawdown = env_state.get('cumulative_drawdown', env_state.get('max_drawdown', 0.0))
        
        if cumulative_drawdown > self.max_cumulative_drawdown:
            # Експоненційний штраф за перевищення cumulative просадки
            excess_drawdown = cumulative_drawdown - self.max_cumulative_drawdown
            penalty = np.exp(excess_drawdown * 15) - 1  # Агресивний штраф
            return -penalty * abs(self.weight)
        elif cumulative_drawdown > self.max_cumulative_drawdown * 0.7:  # 14% при ліміті 20%
            # Попереджувальний штраф
            warning_penalty = (cumulative_drawdown - self.max_cumulative_drawdown * 0.7) * 10
            return -warning_penalty * abs(self.weight)
        
        return 0.0


class CapitalPreservationReward(BaseRewardScheme):
    """Винагорода за збереження та примноження капіталу."""
    
    def __init__(self, weight: float = 2.0):
        super().__init__(weight)
        self.last_portfolio_value = None
        
    def calculate(self, env_state: Dict) -> float:
        portfolio_value = env_state.get('portfolio_value', 10000.0)
        initial_balance = env_state.get('initial_balance', 10000.0)
        
        # Винагорода за збереження капіталу вище початкового рівня
        if portfolio_value > initial_balance:
            preservation_ratio = portfolio_value / initial_balance
            if preservation_ratio > 1.5:  # >150% збереження
                reward = 3.0 * self.weight
            elif preservation_ratio > 1.2:  # >120% збереження
                reward = 2.0 * self.weight
            elif preservation_ratio > 1.1:  # >110% збереження
                reward = 1.0 * self.weight
            else:  # >100% збереження
                reward = 0.5 * self.weight
        elif portfolio_value > initial_balance * 0.9:  # >90% збереження
            reward = 0.1 * self.weight
        else:  # <90% збереження - штраф
            loss_ratio = (initial_balance - portfolio_value) / initial_balance
            reward = -loss_ratio * 5.0 * abs(self.weight)
            
        return np.clip(reward, -5.0, 10.0)


class PortfolioVolatilityPenalty(BaseRewardScheme):
    """Штраф за волатільність cumulative портфеля."""
    
    def __init__(self, weight: float = -1.0, window: int = 10):
        super().__init__(weight)
        self.window = window
        
    def calculate(self, env_state: Dict) -> float:
        cumulative_history = env_state.get('cumulative_portfolio_history', [])
        portfolio_value = env_state.get('portfolio_value', 10000.0)
        
        if len(cumulative_history) < self.window:
            return 0.0
        
        # Аналізуємо волатільність останніх значень портфеля
        recent_values = cumulative_history[-self.window:] + [portfolio_value]
        returns = np.diff(recent_values) / recent_values[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # Штраф за високу волатільність
        if volatility > 0.1:  # >10% волатільність між епізодами
            penalty = volatility * 20.0 * abs(self.weight)
            return -min(penalty, 5.0)
        
        return 0.0


class ConsistentGrowthReward(BaseRewardScheme):
    """Винагорода за консистентне зростання портфеля."""
    
    def __init__(self, weight: float = 1.5, window: int = 5):
        super().__init__(weight)
        self.window = window
        
    def calculate(self, env_state: Dict) -> float:
        cumulative_history = env_state.get('cumulative_portfolio_history', [])
        portfolio_value = env_state.get('portfolio_value', 10000.0)
        
        if len(cumulative_history) < self.window:
            return 0.0
        
        # Аналізуємо тренд зростання
        recent_values = cumulative_history[-self.window:] + [portfolio_value]
        
        # Перевіряємо чи є загальний upward trend
        positive_changes = 0
        total_changes = len(recent_values) - 1
        
        for i in range(1, len(recent_values)):
            if recent_values[i] > recent_values[i-1]:
                positive_changes += 1
        
        consistency_ratio = positive_changes / total_changes if total_changes > 0 else 0
        
        # Винагорода за консистентність
        if consistency_ratio >= 0.8:  # 80%+ позитивних змін
            return 2.0 * self.weight
        elif consistency_ratio >= 0.6:  # 60%+ позитивних змін
            return 1.0 * self.weight
        elif consistency_ratio >= 0.4:  # 40%+ позитивних змін
            return 0.5 * self.weight
        
        return 0.0


def create_cumulative_growth_reward_scheme() -> CompositeRewardScheme:
    """
    НОВА СХЕМА для портфельної континуальності:
    - Фокус на CUMULATIVE зростання портфеля через епізоди
    - Винагороди за збереження та примноження капіталу
    - Жорсткий контроль cumulative просадки
    - Заохочення довгострокового мислення
    """
    schemes = [
        # СПРОЩЕНА схема для стабільного навчання
        CumulativeGrowthReward(weight=2.0),  # Основна винагорода за зростання
        
        # Помірний контроль просадки
        CumulativeDrawdownPenalty(weight=-1.0, max_cumulative_drawdown=0.25),  # 25% максимум
        
        # Винагорода за торгову активність
        ExplorationReward(weight=1.0, target_trades_per_episode=20),  # Заохочення активності
    ]
    
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=True)
    return composite


def create_risk_adjusted_reward_scheme() -> CompositeRewardScheme:
    """
    РИЗИК-ЗБАЛАНСОВАНА схема винагород для стабільного навчання:
    - Сильний акцент на контроль ризиків та просадки
    - Помірні винагороди за прибуток з урахуванням ризику
    - Заохочення консистентності та стабільності
    - Запобігання волатильній поведінці
    """
    schemes = [
        # ГОЛОВНИЙ КОМПОНЕНТ: Збалансована винагорода за прибуток
        TotalReturnReward(weight=3.0),  # ЗБІЛЬШЕНО для кращих позитивних сигналів
        
        # ЖОРСТКИЙ контроль просадки для досягнення цільових 20%
        DrawdownPenalty(weight=-3.0, max_drawdown_threshold=0.08),  # ЗБІЛЬШЕНО штраф, ЗМЕНШЕНО поріг до 8%
        
        # ВИНАГОРОДА за коефіцієнт Шарпа (ризик-скорегована доходність)
        SharpeRatioReward(weight=1.0, window=50),  # ЗМЕНШЕНО вагу, ЗБІЛЬШЕНО вікно для стабільності
        
        # М'ЯКА винагорода за консистентність
        ConsistencyReward(weight=0.5, window=30),  # ЗМЕНШЕНО для менших вимог під час навчання
        
        # ЛЕГКИЙ штраф за волатільність (тільки при екстремальних значеннях)
        VolatilityPenalty(weight=-0.3, window=20),  # ЗМЕНШЕНО для заохочення експлорації
        
        # ВИНАГОРОДА за якість угод з м'якими вимогами
        TradeQualityReward(weight=1.0, min_trades=1),  # ЗМЕНШЕНО мінімум до 1 угоди для навчання
        
        # ДУЖЕ М'ЯКІ штрафи за збитки (заохочуємо експлорацію)
        LossTradesPenalty(weight=-0.5),  # СИЛЬНО ЗМЕНШЕНО для кращого навчання
        
        # ВИМКНЕНО: М'який штраф за погіршення показників (занадто агресивний під час навчання)
        # PerformanceDeclineReward(weight=-0.2, decline_threshold=0.20),  # ТИМЧАСОВО ВИМКНЕНО
        
        # АГРЕСИВНА винагорода за торгову активність та експлорацію 
        ExplorationReward(weight=3.0, target_trades_per_episode=20),  # ЗБІЛЬШЕНО для форсованої активності
    ]
    
    composite = CompositeRewardScheme(schemes, enable_dynamic_scaling=True)  # УВІМКНЕНО для стабілізації
    return composite