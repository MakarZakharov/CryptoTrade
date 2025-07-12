"""
Callbacks для процесу навчання STAS_ML агентів.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback


class TradingCallback(BaseCallback):
    """Callback для моніторингу торгових метрик."""
    
    def __init__(self, log_dir: str, experiment_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_drawdowns = []
        self.episode_win_rates = []
        self.episode_trades = []
        
        # Структурований вивід метрик
        self.analysis_data = []
        self.last_report_step = 0
        self.report_interval = 100  # Кожні 100 кроків
        
        # Створюємо директорію для логів
        os.makedirs(log_dir, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Викликається на кожному кроці."""
        current_step = self.num_timesteps
        
        # Отримуємо інформацію з середовища
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Збираємо дані для структурованого виводу
            if 'portfolio_value' in info and 'total_return' in info:
                step_data = {
                    'step': current_step,
                    'portfolio_value': info.get('portfolio_value', 0),
                    'total_return': info.get('total_return', 0),
                    'max_drawdown': info.get('max_drawdown', 0),
                    'total_trades': info.get('total_trades', 0),
                    'win_rate': info.get('win_rate', 0),
                    'current_price': info.get('current_price', 0)
                }
                self.analysis_data.append(step_data)
            
            # Записуємо метрики в tensorboard
            if 'portfolio_value' in info:
                self.logger.record('trading/portfolio_value', info['portfolio_value'])
            if 'total_return' in info:
                self.logger.record('trading/total_return', info['total_return'])
            if 'max_drawdown' in info:
                self.logger.record('trading/max_drawdown', info['max_drawdown'])
            if 'win_rate' in info:
                self.logger.record('trading/win_rate', info['win_rate'])
            if 'total_trades' in info:
                self.logger.record('trading/total_trades', info['total_trades'])
            
            # Структурований вивід кожні report_interval кроків
            if current_step - self.last_report_step >= self.report_interval:
                self._print_structured_report()
                self.last_report_step = current_step
        
        return True
    
    def _print_structured_report(self):
        """Виводить структурований звіт по етапах навчання."""
        if not self.analysis_data:
            return
        
        current_data = self.analysis_data[-1]
        step = current_data['step']
        
        print(f"\n{'='*60}")
        print(f"📊 ЗВІТ ПО НАВЧАННЮ - Крок {step:,}")
        print(f"{'='*60}")
        
        # Основні метрики
        profit_percent = current_data['total_return'] * 100
        drawdown_percent = current_data['max_drawdown'] * 100
        total_trades = current_data['total_trades']
        win_rate_percent = current_data['win_rate'] * 100
        
        # Визначаємо статус прибутковості
        if profit_percent > 0:
            profit_status = "🟢 ПРИБУТОК"
        elif profit_percent < -5:
            profit_status = "🔴 ЗБИТОК"
        else:
            profit_status = "🟡 БЕЗЗБИТКОВІСТЬ"
        
        # Визначаємо рівень просадки
        if drawdown_percent < 5:
            drawdown_status = "🟢 НИЗЬКА"
        elif drawdown_percent < 15:
            drawdown_status = "🟡 ПОМІРНА"
        else:
            drawdown_status = "🔴 ВИСОКА"
        
        print(f"💰 ПРИБУТКОВІСТЬ:")
        print(f"   Загальна доходність: {profit_percent:+.2f}% ({profit_status})")
        print(f"   Поточна вартість портфеля: ${current_data['portfolio_value']:,.2f}")
        
        print(f"\n📉 РИЗИКИ:")
        print(f"   Максимальна просадка: {drawdown_percent:.2f}% ({drawdown_status})")
        
        print(f"\n📈 ТОРГОВА АКТИВНІСТЬ:")
        print(f"   Кількість угод: {total_trades}")
        print(f"   Відсоток прибуткових угод: {win_rate_percent:.1f}%")
        
        # Аналіз ефективності за останні кроки
        if len(self.analysis_data) >= 10:
            recent_data = self.analysis_data[-10:]
            recent_returns = [d['total_return'] for d in recent_data]
            recent_trend = recent_returns[-1] - recent_returns[0]
            
            if recent_trend > 0.01:
                trend_status = "📈 ЗРОСТАЮЧИЙ"
            elif recent_trend < -0.01:
                trend_status = "📉 СПАДНИЙ"
            else:
                trend_status = "➡️ СТАБІЛЬНИЙ"
            
            print(f"\n📊 ТРЕНД (останні {len(recent_data)} кроків):")
            print(f"   Напрямок: {trend_status}")
            print(f"   Зміна доходності: {recent_trend*100:+.2f}%")
        
        print(f"\n💡 РЕКОМЕНДАЦІЇ:")
        
        # Рекомендації на основі метрик
        recommendations = []
        
        if drawdown_percent > 20:
            recommendations.append("⚠️ Висока просадка - розгляньте зменшення розміру позицій")
        
        if total_trades < step // 100:
            recommendations.append("🔄 Низька торгова активність - агент може бути занадто консервативним")
        
        if win_rate_percent < 40 and total_trades > 10:
            recommendations.append("🎯 Низький винрейт - потрібно покращити стратегію входу/виходу")
        
        if profit_percent > 10 and drawdown_percent < 10:
            recommendations.append("✅ Відмінні результати - продовжуйте навчання")
        
        if not recommendations:
            recommendations.append("📊 Результати в межах норми - продовжуйте спостереження")
        
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"{'='*60}\n")
        
        # Зберігаємо звіт у файл
        self._save_report_to_file()
    
    def _save_report_to_file(self):
        """Зберігає детальний звіт у CSV файл."""
        if not self.analysis_data:
            return
        
        # Створюємо DataFrame з усіх зібраних даних
        df = pd.DataFrame(self.analysis_data)
        
        # Додаємо розраховані колонки
        df['profit_percent'] = df['total_return'] * 100
        df['drawdown_percent'] = df['max_drawdown'] * 100
        df['win_rate_percent'] = df['win_rate'] * 100
        
        # Зберігаємо у файл
        report_path = os.path.join(self.log_dir, 'training_report.csv')
        df.to_csv(report_path, index=False)
        
        # Також створюємо підсумковий звіт
        if len(df) > 0:
            summary = {
                'final_step': df['step'].iloc[-1],
                'final_profit_percent': df['profit_percent'].iloc[-1],
                'max_drawdown_percent': df['drawdown_percent'].max(),
                'total_trades': df['total_trades'].iloc[-1],
                'final_win_rate_percent': df['win_rate_percent'].iloc[-1],
                'best_return_percent': df['profit_percent'].max(),
                'worst_drawdown_percent': df['drawdown_percent'].max()
            }
            
            summary_df = pd.DataFrame([summary])
            summary_path = os.path.join(self.log_dir, 'training_summary.csv')
            summary_df.to_csv(summary_path, index=False)
    
    def _on_episode_end(self) -> None:
        """Викликається в кінці епізоду."""
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Зберігаємо метрики епізоду
            self.episode_returns.append(info.get('total_return', 0))
            self.episode_drawdowns.append(info.get('max_drawdown', 0))
            self.episode_win_rates.append(info.get('win_rate', 0))
            self.episode_trades.append(info.get('total_trades', 0))
            
            # Записуємо агреговані метрики
            if len(self.episode_returns) > 0:
                self.logger.record('episode/mean_return', np.mean(self.episode_returns[-100:]))
                self.logger.record('episode/mean_drawdown', np.mean(self.episode_drawdowns[-100:]))
                self.logger.record('episode/mean_win_rate', np.mean(self.episode_win_rates[-100:]))
                self.logger.record('episode/mean_trades', np.mean(self.episode_trades[-100:]))


class TensorboardCallback(BaseCallback):
    """Callback для розширеного логування в Tensorboard."""
    
    def __init__(self, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.step_count = 0
        
        # Створюємо директорію
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Логування на кожному кроці."""
        self.step_count += 1
        
        # Логуємо кожні 1000 кроків
        if self.step_count % 1000 == 0:
            # Отримуємо винагороди
            if 'rewards' in self.locals:
                rewards = self.locals['rewards']
                if len(rewards) > 0:
                    self.logger.record('reward/mean_reward', np.mean(rewards))
                    self.logger.record('reward/max_reward', np.max(rewards))
                    self.logger.record('reward/min_reward', np.min(rewards))
            
            # Логуємо дії агента
            if 'actions' in self.locals:
                actions = self.locals['actions']
                if len(actions) > 0:
                    self.logger.record('action/mean_action', np.mean(actions))
                    self.logger.record('action/std_action', np.std(actions))
        
        return True