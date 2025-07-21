"""Модуль визуализации результатов DRL бектестинга."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..utils import DRLLogger


class BacktestVisualizer:
    """
    Система визуализации результатов бектестинга DRL агентов.
    
    Создает различные графики и диаграммы для анализа
    производительности обученных агентов.
    """
    
    def __init__(self, logger: Optional[DRLLogger] = None):
        """
        Инициализация визуализатора.
        
        Args:
            logger: логгер
        """
        self.logger = logger or DRLLogger("backtest_visualizer")
        
        # Настройка стиля графиков
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Настройки matplotlib для работы в разных средах
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Fallback для серверной среды
            except:
                pass
        
        self.logger.info("Visualizer инициализирован")
    
    def create_comprehensive_report(
        self,
        backtest_results: Dict[str, Any],
        episode_data: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        show_plots: bool = True
    ) -> str:
        """
        Создание комплексного отчета с графиками.
        
        Args:
            backtest_results: результаты бектеста
            episode_data: данные эпизода
            save_path: путь для сохранения
            show_plots: показывать графики
            
        Returns:
            Путь к сохраненным графикам
        """
        self.logger.info("Создание комплексного визуального отчета...")
        
        # Определяем путь сохранения
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"CryptoTrade/ai/DRL/evaluation/results/visual_report_{timestamp}"
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем различные графики
        if episode_data:
            self._plot_portfolio_evolution(episode_data, save_dir, show_plots)
            self._plot_price_and_actions(episode_data, save_dir, show_plots)
            self._plot_rewards_distribution(episode_data, save_dir, show_plots)
            self._plot_drawdown_analysis(episode_data, save_dir, show_plots)
        
        # Метрики производительности
        self._plot_performance_metrics(backtest_results, save_dir, show_plots)
        
        # Сравнительный анализ
        self._plot_comparative_analysis(backtest_results, save_dir, show_plots)
        
        self.logger.info(f"Визуальный отчет создан: {save_dir}")
        return str(save_dir)
    
    def _plot_portfolio_evolution(
        self,
        episode_data: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """График эволюции портфеля."""
        try:
            portfolio_values = episode_data.get('portfolio_values', [])
            prices = episode_data.get('prices', [])
            
            if not portfolio_values or not prices:
                self.logger.warning("Недостаточно данных для графика портфеля")
                return
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # График стоимости портфеля
            steps = range(len(portfolio_values))
            ax1.plot(steps, portfolio_values, label='Стоимость портфеля', color='blue', linewidth=2)
            ax1.axhline(y=portfolio_values[0], color='gray', linestyle='--', alpha=0.7, label='Начальная стоимость')
            ax1.set_title('Эволюция стоимости портфеля', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Стоимость ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График цены актива
            ax2.plot(steps, prices, label='Цена актива', color='orange', linewidth=2)
            ax2.set_title('Цена базового актива', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Шаги', fontsize=12)
            ax2.set_ylabel('Цена ($)', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "portfolio_evolution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания графика портфеля: {e}")
    
    def _plot_price_and_actions(
        self,
        episode_data: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """График цены с отмеченными действиями агента."""
        try:
            prices = episode_data.get('prices', [])
            actions = episode_data.get('actions', [])
            
            if not prices or not actions:
                self.logger.warning("Недостаточно данных для графика действий")
                return
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            steps = range(len(prices))
            
            # График цены
            ax.plot(steps, prices, label='Цена', color='black', linewidth=1, alpha=0.8)
            
            # Отмечаем действия агента
            buy_steps = []
            sell_steps = []
            hold_steps = []
            
            for i, action in enumerate(actions):
                if isinstance(action, (int, float)):
                    if action > 0.1:  # Покупка
                        buy_steps.append(i)
                    elif action < -0.1:  # Продажа
                        sell_steps.append(i)
                    else:  # Удержание
                        hold_steps.append(i)
                elif hasattr(action, '__len__'):  # Массив действий
                    # Для дискретных действий
                    if len(action) > 0:
                        if action[0] == 1:  # Покупка
                            buy_steps.append(i)
                        elif action[0] == 2:  # Продажа
                            sell_steps.append(i)
                        else:  # Удержание
                            hold_steps.append(i)
            
            # Отмечаем сигналы на графике
            if buy_steps:
                buy_prices = [prices[i] for i in buy_steps if i < len(prices)]
                ax.scatter(buy_steps[:len(buy_prices)], buy_prices, 
                          color='green', marker='^', s=50, label='Покупка', alpha=0.7)
            
            if sell_steps:
                sell_prices = [prices[i] for i in sell_steps if i < len(prices)]
                ax.scatter(sell_steps[:len(sell_prices)], sell_prices, 
                          color='red', marker='v', s=50, label='Продажа', alpha=0.7)
            
            ax.set_title('Цена актива и действия агента', fontsize=16, fontweight='bold')
            ax.set_xlabel('Шаги', fontsize=12)
            ax.set_ylabel('Цена ($)', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "price_and_actions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания графика действий: {e}")
    
    def _plot_rewards_distribution(
        self,
        episode_data: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """График распределения наград."""
        try:
            rewards = episode_data.get('rewards', [])
            
            if not rewards:
                self.logger.warning("Недостаточно данных для графика наград")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # График наград по времени
            ax1.plot(rewards, color='purple', alpha=0.7)
            ax1.set_title('Награды по времени')
            ax1.set_xlabel('Шаги')
            ax1.set_ylabel('Награда')
            ax1.grid(True, alpha=0.3)
            
            # Гистограмма распределения наград
            ax2.hist(rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_title('Распределение наград')
            ax2.set_xlabel('Величина награды')
            ax2.set_ylabel('Частота')
            ax2.grid(True, alpha=0.3)
            
            # Кумулятивные награды
            cumulative_rewards = np.cumsum(rewards)
            ax3.plot(cumulative_rewards, color='green')
            ax3.set_title('Кумулятивные награды')
            ax3.set_xlabel('Шаги')
            ax3.set_ylabel('Кумулятивная награда')
            ax3.grid(True, alpha=0.3)
            
            # Box plot наград по периодам
            reward_chunks = [rewards[i:i+len(rewards)//10] 
                            for i in range(0, len(rewards), len(rewards)//10)]
            ax4.boxplot(reward_chunks, labels=[f'P{i+1}' for i in range(len(reward_chunks))])
            ax4.set_title('Распределение наград по периодам')
            ax4.set_xlabel('Период')
            ax4.set_ylabel('Награда')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "rewards_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания графика наград: {e}")
    
    def _plot_drawdown_analysis(
        self,
        episode_data: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """Анализ просадок."""
        try:
            portfolio_values = episode_data.get('portfolio_values', [])
            
            if not portfolio_values:
                self.logger.warning("Недостаточно данных для анализа просадок")
                return
            
            # Расчет просадок
            portfolio_array = np.array(portfolio_values)
            cummax = np.maximum.accumulate(portfolio_array)
            drawdown = (cummax - portfolio_array) / cummax * 100
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # График просадки
            ax1.fill_between(range(len(drawdown)), drawdown, 0, 
                           alpha=0.5, color='red', label='Просадка')
            ax1.set_title('Анализ просадки портфеля', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Просадка (%)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.invert_yaxis()  # Инвертируем ось Y для просадки
            
            # График стоимости портфеля с отмеченными максимумами
            ax2.plot(portfolio_values, color='blue', label='Стоимость портфеля')
            ax2.plot(cummax, color='green', linestyle='--', alpha=0.7, label='Максимальная стоимость')
            ax2.set_title('Стоимость портфеля и пиковые значения')
            ax2.set_xlabel('Шаги')
            ax2.set_ylabel('Стоимость ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "drawdown_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания графика просадок: {e}")
    
    def _plot_performance_metrics(
        self,
        backtest_results: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """График ключевых метрик производительности."""
        try:
            if 'performance' not in backtest_results:
                self.logger.warning("Нет данных о производительности")
                return
            
            perf = backtest_results['performance']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            
            # Барчарт доходности
            returns_data = [
                perf.get('total_return_pct', 0),
                perf.get('buy_hold_return_pct', 0)
            ]
            ax1.bar(['DRL Agent', 'Buy & Hold'], returns_data, 
                   color=['blue', 'orange'], alpha=0.7)
            ax1.set_title('Сравнение доходности')
            ax1.set_ylabel('Доходность (%)')
            ax1.grid(True, alpha=0.3)
            
            # Радарная диаграмма ключевых метрик
            metrics = ['Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Volatility']
            values = [
                perf.get('sharpe_ratio', 0),
                perf.get('total_return_pct', 0) / 100,  # Нормализуем
                1 - perf.get('max_drawdown_pct', 0) / 100,  # Инвертируем (меньше = лучше)
                1 - min(perf.get('volatility', 0) / 100, 1)  # Нормализуем и инвертируем
            ]
            
            # Простой столбчатый график вместо радарного
            ax2.barh(metrics, values, color='green', alpha=0.7)
            ax2.set_title('Ключевые метрики')
            ax2.set_xlabel('Нормализованное значение')
            
            # Торговые метрики
            if 'trading' in backtest_results:
                trading = backtest_results['trading']
                trade_metrics = ['Total Trades', 'Win Rate (%)', 'Profit Factor']
                trade_values = [
                    trading.get('total_trades', 0),
                    trading.get('win_rate', 0) * 100,
                    trading.get('profit_factor', 0)
                ]
                
                ax3.bar(trade_metrics, trade_values, color='purple', alpha=0.7)
                ax3.set_title('Торговые метрики')
                ax3.tick_params(axis='x', rotation=45)
            
            # Pie chart распределения сделок
            if 'trading' in backtest_results:
                trading = backtest_results['trading']
                profitable = trading.get('profitable_trades', 0)
                losing = trading.get('losing_trades', 0)
                
                if profitable > 0 or losing > 0:
                    sizes = [profitable, losing]
                    labels = ['Прибыльные', 'Убыточные']
                    colors = ['green', 'red']
                    
                    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                           startangle=90, alpha=0.7)
                    ax4.set_title('Распределение сделок')
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "performance_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания графика метрик: {e}")
    
    def _plot_comparative_analysis(
        self,
        backtest_results: Dict[str, Any],
        save_dir: Path,
        show_plots: bool
    ):
        """Сравнительный анализ с базовыми стратегиями."""
        try:
            if 'performance' not in backtest_results:
                return
            
            perf = backtest_results['performance']
            
            # Данные для сравнения
            strategies = ['DRL Agent', 'Buy & Hold', 'Random']
            returns = [
                perf.get('total_return_pct', 0),
                perf.get('buy_hold_return_pct', 0),
                0  # Случайная стратегия как baseline
            ]
            
            sharpe_ratios = [
                perf.get('sharpe_ratio', 0),
                perf.get('buy_hold_return_pct', 0) / max(perf.get('volatility', 1), 0.01),  # Приблизительный Sharpe для B&H
                0
            ]
            
            max_drawdowns = [
                perf.get('max_drawdown_pct', 0),
                abs(perf.get('buy_hold_return_pct', 0)) * 0.3,  # Примерная оценка
                50  # Высокая просадка для случайной стратегии
            ]
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Сравнение доходности
            bars1 = ax1.bar(strategies, returns, color=['blue', 'green', 'gray'], alpha=0.7)
            ax1.set_title('Сравнение доходности')
            ax1.set_ylabel('Доходность (%)')
            ax1.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, value in zip(bars1, returns):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(returns)*0.01,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            # Сравнение Sharpe ratio
            bars2 = ax2.bar(strategies, sharpe_ratios, color=['blue', 'green', 'gray'], alpha=0.7)
            ax2.set_title('Сравнение Sharpe Ratio')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, sharpe_ratios):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sharpe_ratios)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
            
            # Сравнение максимальной просадки
            bars3 = ax3.bar(strategies, max_drawdowns, color=['blue', 'green', 'gray'], alpha=0.7)
            ax3.set_title('Сравнение макс. просадки')
            ax3.set_ylabel('Максимальная просадка (%)')
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, max_drawdowns):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(max_drawdowns)*0.01,
                        f'{value:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Сохранение
            save_path = save_dir / "comparative_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.error(f"Ошибка создания сравнительного анализа: {e}")
    
    def create_summary_dashboard(
        self,
        backtest_results: Dict[str, Any],
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> str:
        """
        Создание итогового дашборда с ключевыми метриками.
        
        Args:
            backtest_results: результаты бектеста
            save_path: путь для сохранения
            show_plot: показать график
            
        Returns:
            Путь к сохраненному дашборду
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"CryptoTrade/ai/DRL/evaluation/results/dashboard_{timestamp}.png"
        
        # Создаем дашборд
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Основные метрики (большой блок)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        self._create_main_metrics_panel(ax_main, backtest_results)
        
        # Доходность vs базовые стратегии
        ax_returns = fig.add_subplot(gs[0, 2])
        self._create_returns_comparison(ax_returns, backtest_results)
        
        # Торговые метрики
        ax_trading = fig.add_subplot(gs[0, 3])
        self._create_trading_metrics(ax_trading, backtest_results)
        
        # Risk metrics
        ax_risk = fig.add_subplot(gs[1, 2])
        self._create_risk_metrics(ax_risk, backtest_results)
        
        # Награды summary
        ax_rewards = fig.add_subplot(gs[1, 3])
        self._create_rewards_summary(ax_rewards, backtest_results)
        
        # Итоговая сводка (нижняя полоса)
        ax_summary = fig.add_subplot(gs[2, :])
        self._create_final_summary(ax_summary, backtest_results)
        
        # Общий заголовок
        fig.suptitle('DRL Agent Backtest Dashboard', fontsize=20, fontweight='bold')
        
        # Сохранение
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        self.logger.info(f"Дашборд сохранен: {save_path}")
        return save_path
    
    def _create_main_metrics_panel(self, ax, backtest_results):
        """Основная панель с ключевыми метриками."""
        ax.axis('off')
        
        # Получаем метрики
        perf = backtest_results.get('performance', {})
        meta = backtest_results.get('metadata', {})
        
        # Создаем текст с метриками
        metrics_text = f"""
        🎯 ОСНОВНЫЕ РЕЗУЛЬТАТЫ
        ────────────────────────────
        
        💰 Итоговая доходность: {perf.get('total_return_pct', 0):.2f}%
        📊 Коэффициент Шарпа: {perf.get('sharpe_ratio', 0):.2f}
        📉 Максимальная просадка: {perf.get('max_drawdown_pct', 0):.2f}%
        
        🚀 vs Buy & Hold: {perf.get('alpha', 0)*100:.2f}%
        
        ⚡ Символ: {meta.get('symbol', 'N/A')}
        ⏰ Таймфрейм: {meta.get('timeframe', 'N/A')}
        📝 Всего шагов: {meta.get('total_steps', 0):,}
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    
    def _create_returns_comparison(self, ax, backtest_results):
        """Сравнение доходности."""
        perf = backtest_results.get('performance', {})
        
        returns = [
            perf.get('total_return_pct', 0),
            perf.get('buy_hold_return_pct', 0)
        ]
        
        bars = ax.bar(['DRL', 'B&H'], returns, color=['blue', 'orange'], alpha=0.7)
        ax.set_title('Доходность (%)')
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, returns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(abs(max(returns)), abs(min(returns)))*0.05,
                   f'{value:.1f}%', ha='center', va='bottom' if value >= 0 else 'top')
    
    def _create_trading_metrics(self, ax, backtest_results):
        """Торговые метрики."""
        trading = backtest_results.get('trading', {})
        
        win_rate = trading.get('win_rate', 0) * 100
        profit_factor = trading.get('profit_factor', 0)
        
        metrics = ['Win Rate (%)', 'Profit Factor']
        values = [win_rate, profit_factor]
        
        bars = ax.bar(metrics, values, color='green', alpha=0.7)
        ax.set_title('Торговля')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.05,
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _create_risk_metrics(self, ax, backtest_results):
        """Метрики риска."""
        perf = backtest_results.get('performance', {})
        
        volatility = perf.get('volatility', 0)
        max_dd = perf.get('max_drawdown_pct', 0)
        
        metrics = ['Volatility', 'Max DD (%)']
        values = [volatility, max_dd]
        
        bars = ax.bar(metrics, values, color='red', alpha=0.7)
        ax.set_title('Риск')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.05,
                   f'{value:.1f}', ha='center', va='bottom')
    
    def _create_rewards_summary(self, ax, backtest_results):
        """Сводка по наградам."""
        rewards = backtest_results.get('rewards', {})
        
        total_reward = rewards.get('total_reward', 0)
        mean_reward = rewards.get('mean_reward', 0)
        
        ax.bar(['Total', 'Mean'], [total_reward, mean_reward], 
               color='purple', alpha=0.7)
        ax.set_title('Награды')
        
    def _create_final_summary(self, ax, backtest_results):
        """Итоговая сводка."""
        ax.axis('off')
        
        summary = backtest_results.get('summary', {})
        
        summary_text = f"""
        🏆 ИТОГОВАЯ ОЦЕНКА: {summary.get('total_return_pct', 0):.2f}% доходности  |  
        📈 Превышение B&H: {summary.get('vs_buy_hold', 0):.2f}%  |  
        ⭐ Sharpe: {summary.get('sharpe_ratio', 0):.2f}  |  
        🛡️ Макс. просадка: {summary.get('max_drawdown_pct', 0):.2f}%  |  
        🎯 Винрейт: {summary.get('win_rate', 0):.1f}%  |  
        💼 Сделок: {summary.get('total_trades', 0)}
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, 
               fontsize=14, ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))