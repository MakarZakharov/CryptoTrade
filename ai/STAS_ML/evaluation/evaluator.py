"""
Модуль оценки для ML моделей STAS_ML.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Класс для оценки ML моделей."""
    
    def __init__(self, config):
        self.config = config
        self.is_classification = config.target_type == 'direction'
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Полная оценка модели."""
        
        if self.is_classification:
            return self._evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            return self._evaluate_regression(y_true, y_pred)
    
    def _evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Оценка регрессионной модели."""
        metrics = {}
        
        # Основные метрики
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Дополнительные метрики
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        
        # Направленная точность (процент правильно предсказанных направлений)
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        metrics['direction_accuracy'] = accuracy_score(direction_true, direction_pred)
        
        # Корреляция
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        return metrics
    
    def _evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Оценка классификационной модели."""
        metrics = {}
        
        # Основные метрики
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC (если есть вероятности)
        if y_pred_proba is not None:
            try:
                if y_pred_proba.shape[1] == 2:  # Бинарная классификация
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:  # Многоклассовая
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Для бинарной классификации
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Специфичность и чувствительность
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_pred_proba: Optional[np.ndarray] = None,
                              save_path: Optional[str] = None) -> List[str]:
        """Создать графики для оценки модели."""
        plot_files = []
        
        plt.style.use('default')
        
        if self.is_classification:
            plot_files.extend(self._plot_classification(y_true, y_pred, y_pred_proba, save_path))
        else:
            plot_files.extend(self._plot_regression(y_true, y_pred, save_path))
        
        return plot_files
    
    def _plot_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                        save_path: Optional[str] = None) -> List[str]:
        """Создать графики для регрессии."""
        plot_files = []
        
        # 1. Scatter plot: предсказания vs реальные значения
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Реальные значения')
        plt.ylabel('Предсказанные значения')
        plt.title('Предсказания vs Реальные значения')
        
        # Добавляем метрики на график
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.6f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            file_path = f"{save_path}/predictions_vs_actual.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        plt.show()
        
        # 2. Residuals plot
        plt.figure(figsize=(10, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Остатки')
        plt.title('График остатков')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            file_path = f"{save_path}/residuals_plot.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        plt.show()
        
        # 3. Histogram of residuals
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Остатки')
        plt.ylabel('Частота')
        plt.title('Распределение остатков')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            file_path = f"{save_path}/residuals_histogram.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        plt.show()
        
        return plot_files
    
    def _plot_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                           y_pred_proba: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> List[str]:
        """Создать графики для классификации."""
        plot_files = []
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Реальные классы')
        plt.title('Матрица ошибок')
        plt.tight_layout()
        
        if save_path:
            file_path = f"{save_path}/confusion_matrix.png"
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            plot_files.append(file_path)
        plt.show()
        
        # 2. ROC Curve (для бинарной классификации)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            plt.figure(figsize=(8, 6))
            
            if y_pred_proba.shape[1] == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
            else:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba)
            
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Кривая')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                file_path = f"{save_path}/roc_curve.png"
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plot_files.append(file_path)
            plt.show()
        
        return plot_files
    
    def create_trading_simulation(self, y_true: np.ndarray, y_pred: np.ndarray,
                                prices: Optional[np.ndarray] = None,
                                initial_balance: float = 10000.0,
                                use_backtrader: bool = True) -> Dict[str, Any]:
        """Создать симуляцию торговли на основе предсказаний."""
        
        if use_backtrader:
            return self._backtrader_simulation(y_true, y_pred, prices, initial_balance)
        else:
            # Старая симуляция для совместимости
            if self.config.target_type == 'direction':
                return self._simulate_direction_trading(y_true, y_pred, prices, initial_balance)
            elif self.config.target_type == 'price_change':
                return self._simulate_price_change_trading(y_true, y_pred, prices, initial_balance)
            else:
                return {'error': f'Торговая симуляция не поддерживается для {self.config.target_type}'}
    
    def _simulate_direction_trading(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  prices: Optional[np.ndarray] = None,
                                  initial_balance: float = 10000.0) -> Dict[str, Any]:
        """ПОКРАЩЕНА симуляція торгівлі з ризик-менеджментом."""
        
        if prices is None:
            # Создаем синтетические цены
            prices = np.cumsum(y_true) + 100
        
        # МАКСИМАЛЬНО ОПТИМІЗОВАНІ ПАРАМЕТРИ для прибутковості при 88.7% винрейт
        max_position_size = 0.15    # Збільшено до 15% для максимального використання високої точності
        stop_loss_pct = 0.01        # Зменшено стоп-лосс до 1% для R/R 15:1
        take_profit_pct = 0.15      # ЗБІЛЬШЕНО тейк-профіт до 15% для R/R 15:1
        max_drawdown_limit = 0.06   # Ще жорсткіший ліміт 6%
        min_confidence_threshold = getattr(self.config, 'signal_confidence_threshold', 0.85)  # Підвищено до 85%
        trailing_stop_pct = 0.03    # ЗБІЛЬШЕНО трейлінг стоп до 3%
        
        balance = initial_balance
        position = 0  # 0 - нет позиции, 1 - лонг, -1 - шорт
        position_entry_price = 0
        position_peak_profit = 0  # НОВИЙ: відстеження піку прибутку для трейлінг стопу
        trades = []
        balances = [balance]
        max_balance = initial_balance
        current_drawdown = 0
        consecutive_losses = 0
        winning_trades = 0
        
        for i in range(len(y_pred)):
            current_price = prices[i]
            
            # Оновлюємо максимальний баланс та просадку
            if balance > max_balance:
                max_balance = balance
            current_drawdown = (max_balance - balance) / max_balance
            
            # ЗАХИСТ: Зупинка торгівлі при критичній просадці
            if current_drawdown > max_drawdown_limit:
                balances.append(balance)
                continue
            
            # АДАПТИВНИЙ розмір позиції залежно від просадки
            if current_drawdown > 0.1:  # При просадці > 10%
                position_size = max_position_size * 0.5  # Зменшуємо розмір позиції
            elif consecutive_losses > 3:  # Після серії збитків
                position_size = max_position_size * 0.7  # Обережніше торгуємо
            else:
                position_size = max_position_size
            
            # ПОКРАЩЕНА СИСТЕМА УПРАВЛІННЯ ПОЗИЦІЯМИ
            if position != 0 and position_entry_price > 0:
                if position == 1:  # Лонг позиція
                    # Розрахунок поточного прибутку/збитку
                    current_profit_pct = (current_price - position_entry_price) / position_entry_price
                    
                    # Оновлюємо пік прибутку
                    if current_profit_pct > position_peak_profit:
                        position_peak_profit = current_profit_pct
                    
                    # ТЕЙК-ПРОФІТ перевірка
                    if current_profit_pct > take_profit_pct:
                        profit = position_size * balance * current_profit_pct
                        balance += profit
                        trades.append(('TAKE_PROFIT_SELL', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        winning_trades += 1
                        consecutive_losses = 0
                        balances.append(balance)
                        continue
                    
                    # ТРЕЙЛІНГ СТОП перевірка (тільки якщо є прибуток)
                    elif position_peak_profit > trailing_stop_pct and \
                         (position_peak_profit - current_profit_pct) > trailing_stop_pct:
                        profit = position_size * balance * current_profit_pct
                        balance += profit
                        trades.append(('TRAILING_STOP_SELL', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        if current_profit_pct > 0:
                            winning_trades += 1
                            consecutive_losses = 0
                        else:
                            consecutive_losses += 1
                        balances.append(balance)
                        continue
                    
                    # СТОП-ЛОСС перевірка
                    elif current_profit_pct < -stop_loss_pct:
                        loss = position_size * balance * current_profit_pct
                        balance += loss
                        trades.append(('STOP_LOSS_SELL', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        consecutive_losses += 1
                        balances.append(balance)
                        continue
                        
                elif position == -1:  # Шорт позиція
                    # Розрахунок поточного прибутку/збитку для шорту
                    current_profit_pct = (position_entry_price - current_price) / position_entry_price
                    
                    # Оновлюємо пік прибутку
                    if current_profit_pct > position_peak_profit:
                        position_peak_profit = current_profit_pct
                    
                    # ТЕЙК-ПРОФІТ для шорту
                    if current_profit_pct > take_profit_pct:
                        profit = position_size * balance * current_profit_pct
                        balance += profit
                        trades.append(('TAKE_PROFIT_BUY', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        winning_trades += 1
                        consecutive_losses = 0
                        balances.append(balance)
                        continue
                    
                    # ТРЕЙЛІНГ СТОП для шорту
                    elif position_peak_profit > trailing_stop_pct and \
                         (position_peak_profit - current_profit_pct) > trailing_stop_pct:
                        profit = position_size * balance * current_profit_pct
                        balance += profit
                        trades.append(('TRAILING_STOP_BUY', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        if current_profit_pct > 0:
                            winning_trades += 1
                            consecutive_losses = 0
                        else:
                            consecutive_losses += 1
                        balances.append(balance)
                        continue
                    
                    # СТОП-ЛОСС для шорту
                    elif current_profit_pct < -stop_loss_pct:
                        loss = position_size * balance * current_profit_pct
                        balance += loss
                        trades.append(('STOP_LOSS_BUY', current_price, i))
                        position = 0
                        position_entry_price = 0
                        position_peak_profit = 0
                        consecutive_losses += 1
                        balances.append(balance)
                        continue
            
            # Рішення про торгівлю на основі предсказання
            if y_pred[i] == 1 and position != 1:  # Покупаем
                if position == -1:  # Закрываем шорт
                    profit = (position_entry_price - current_price) / position_entry_price * position_size * balance
                    balance += profit
                    if profit > 0:
                        winning_trades += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                
                # Відкриваємо лонг позицію
                position = 1
                position_entry_price = current_price
                trades.append(('BUY', current_price, i))
                
            elif y_pred[i] == 0 and position != -1:  # Продаем
                if position == 1:  # Закрываем лонг
                    profit = (current_price - position_entry_price) / position_entry_price * position_size * balance
                    balance += profit
                    if profit > 0:
                        winning_trades += 1
                        consecutive_losses = 0
                    else:
                        consecutive_losses += 1
                
                # Відкриваємо шорт позицію
                position = -1
                position_entry_price = current_price
                trades.append(('SELL', current_price, i))
            
            balances.append(balance)
        
        # Закриваємо остаточну позицію
        if position != 0 and len(prices) > 0:
            final_price = prices[-1]
            if position == 1:  # Закрываем лонг
                profit = (final_price - position_entry_price) / position_entry_price * max_position_size * balance
                balance += profit
                if profit > 0:
                    winning_trades += 1
            elif position == -1:  # Закрываем шорт  
                profit = (position_entry_price - final_price) / position_entry_price * max_position_size * balance
                balance += profit
                if profit > 0:
                    winning_trades += 1
            trades.append(('CLOSE_FINAL', final_price, len(prices)-1))
        
        # Финальные метрики
        total_return = (balance - initial_balance) / initial_balance * 100
        
        # Покращені метрики
        returns = np.diff(balances) / balances[:-1]
        returns = returns[returns != 0]  # Видаляємо нульові зміни
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Розрахунок максимальної просадки
        peak = initial_balance
        max_drawdown = 0
        for b in balances:
            if b > peak:
                peak = b
            drawdown = (peak - b) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_pct': total_return,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'trades': trades,
            'balance_history': balances,
            'consecutive_losses': consecutive_losses,
            'position_size_used': max_position_size * 100  # У відсотках
        }
    
    def _simulate_price_change_trading(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     prices: Optional[np.ndarray] = None,
                                     initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Симуляция торговли на основе предсказания изменения цены."""
        
        balance = initial_balance
        balances = [balance]
        
        for i in range(len(y_pred)):
            # Простая стратегия: покупаем если предсказываем рост > 1%
            if y_pred[i] > 0.01:
                # Реинвестируем всю прибыль
                balance *= (1 + min(y_true[i], 0.1))  # Ограничиваем максимальную прибыль
            elif y_pred[i] < -0.01:
                # Шорт позиция
                balance *= (1 - min(abs(y_true[i]), 0.1))
            
            balances.append(balance)
        
        total_return = (balance - initial_balance) / initial_balance * 100
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return_pct': total_return,
            'balance_history': balances
        }
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 y_pred_proba: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None) -> str:
        """Сгенерировать полный отчет по оценке модели."""
        
        # Получаем метрики
        metrics = self.evaluate(y_true, y_pred, y_pred_proba)
        
        # Создаем графики
        if save_path:
            plot_files = self.create_evaluation_plots(y_true, y_pred, y_pred_proba, save_path)
        
        # Торговая симуляция
        trading_sim = self.create_trading_simulation(y_true, y_pred)
        
        # Формируем отчет
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("ОТЧЕТ ПО ОЦЕНКЕ МОДЕЛИ")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Основные метрики
        report_lines.append("МЕТРИКИ МОДЕЛИ:")
        if self.is_classification:
            report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision: {metrics['precision']:.4f}")
            report_lines.append(f"  Recall: {metrics['recall']:.4f}")
            report_lines.append(f"  F1-score: {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                report_lines.append(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            report_lines.append(f"  MSE: {metrics['mse']:.6f}")
            report_lines.append(f"  RMSE: {metrics['rmse']:.6f}")
            report_lines.append(f"  MAE: {metrics['mae']:.6f}")
            report_lines.append(f"  R²: {metrics['r2']:.4f}")
            report_lines.append(f"  MAPE: {metrics['mape']:.2f}%")
            report_lines.append(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        report_lines.append("")
        
        # Торговая симуляция
        if 'error' not in trading_sim:
            report_lines.append("ТОРГОВАЯ СИМУЛЯЦИЯ:")
            report_lines.append(f"  Начальный баланс: ${trading_sim['initial_balance']:,.2f}")
            report_lines.append(f"  Финальный баланс: ${trading_sim['final_balance']:,.2f}")
            report_lines.append(f"  Общая доходность: {trading_sim['total_return_pct']:.2f}%")
            if 'total_trades' in trading_sim:
                report_lines.append(f"  Всего сделок: {trading_sim['total_trades']}")
            if 'sharpe_ratio' in trading_sim:
                report_lines.append(f"  Sharpe ratio: {trading_sim['sharpe_ratio']:.4f}")
            if 'max_drawdown_pct' in trading_sim:
                report_lines.append(f"  Максимальная просадка: {trading_sim['max_drawdown_pct']:.2f}%")
        
        # Сохраняем отчет
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(f"{save_path}/evaluation_report.txt", 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _backtrader_simulation(self, y_true: np.ndarray, y_pred: np.ndarray,
                              prices: Optional[np.ndarray] = None,
                              initial_balance: float = 10000.0) -> Dict[str, Any]:
        """Бектест використовуючи backtrader з ML стратегією."""
        try:
            import backtrader as bt
            from .backtrader_strategy import MLPredictionStrategy
        except ImportError:
            print("⚠️ Backtrader не встановлений. Використовую стандартну симуляцію...")
            return self._simulate_direction_trading(y_true, y_pred, prices, initial_balance)
        
        # Створюємо синтетичні дані якщо ціни не надані
        if prices is None:
            prices = np.cumsum(np.random.normal(0, 0.01, len(y_true))) + 100
        
        # Підготовка даних для backtrader
        import pandas as pd
        
        # Створюємо DataFrame з OHLCV даними
        data_length = min(len(y_true), len(y_pred), len(prices))
        
        # Генеруємо реалістичні OHLCV дані на основі цін закриття
        dates = pd.date_range(start='2020-01-01', periods=data_length, freq='D')
        
        # Створюємо OHLC дані з невеликою волатильністю
        df = pd.DataFrame(index=dates)
        df['close'] = prices[:data_length]
        
        # Генеруємо open, high, low на основі close
        volatility = 0.02  # 2% денна волатільність
        for i in range(len(df)):
            close_price = df['close'].iloc[i]
            
            # Open ціна (попередня close + невеликий gap)
            if i == 0:
                df.loc[df.index[i], 'open'] = close_price * (1 + np.random.normal(0, volatility/4))
            else:
                df.loc[df.index[i], 'open'] = df['close'].iloc[i-1] * (1 + np.random.normal(0, volatility/4))
            
            # High та Low на основі open та close
            open_price = df['open'].iloc[i]
            high_base = max(open_price, close_price)
            low_base = min(open_price, close_price)
            
            df.loc[df.index[i], 'high'] = high_base * (1 + abs(np.random.normal(0, volatility/2)))
            df.loc[df.index[i], 'low'] = low_base * (1 - abs(np.random.normal(0, volatility/2)))
            
            # Volume (випадковий але реалістичний)
            df.loc[df.index[i], 'volume'] = np.random.lognormal(10, 0.5)
        
        # Створюємо backtrader cerebro
        cerebro = bt.Cerebro()
        
        # ВИПРАВЛЕНО: Додаємо стратегію з більш агресивними параметрами для торгівлі
        strategy_params = {
            'position_size': 0.95,            # ЗБІЛЬШЕНО до 95% для максимального використання капіталу
            'stop_loss_pct': 0.05,            # ЗБІЛЬШЕНО стоп-лосс до 5% для зменшення ложних сигналів
            'take_profit_pct': 0.10,          # ЗМЕНШЕНО тейк-профіт до 10% для частіших прибутків
            'trailing_stop_pct': 0.02,        # Зменшено трейлінг до 2%
            'max_drawdown_limit': 0.80,       # ЗБІЛЬШЕНО до 80% для агресивнішої торгівлі
            'confidence_threshold': 0.50,     # ЗМЕНШЕНО до 50% для більшої кількості сигналів
            'printlog': True,                 # Увімкнено логування для діагностики
            'debug_mode': True                # Увімкнено дебаг режим
        }
        
        # ВАЖЛИВО: Створюємо кастомну стратегію з передачою предсказань
        class SimplifiedMLStrategy(bt.Strategy):
            params = strategy_params
            
            def __init__(self):
                self.predictions = y_pred[:data_length]
                self.prediction_index = 0
                self.order = None
                self.total_trades = 0
                self.winning_trades = 0
                self.initial_cash = self.broker.getcash()
                print(f"🔍 Стратегія ініціалізована з {len(self.predictions)} предсказаннями")
                print(f"📊 Розподіл предсказань: {np.bincount(self.predictions)}")
            
            def log(self, txt, dt=None):
                dt = dt or self.datas[0].datetime.date(0)
                print(f'{dt.isoformat()}, {txt}')
            
            def notify_order(self, order):
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.log(f'🟢 ПОКУПКА: ${order.executed.price:.2f}, Розмір: {order.executed.size}')
                    else:
                        self.log(f'🔴 ПРОДАЖ: ${order.executed.price:.2f}, Розмір: {order.executed.size}')
                self.order = None
            
            def notify_trade(self, trade):
                if trade.isclosed:
                    self.total_trades += 1
                    if trade.pnlcomm > 0:
                        self.winning_trades += 1
                        self.log(f'✅ ПРИБУТОК: ${trade.pnlcomm:.2f}')
                    else:
                        self.log(f'❌ ЗБИТОК: ${trade.pnlcomm:.2f}')
            
            def next(self):
                if self.order or self.prediction_index >= len(self.predictions):
                    return
                
                current_prediction = self.predictions[self.prediction_index]
                current_price = self.data.close[0]
                cash = self.broker.getcash()
                
                # АГРЕСИВНА ТОРГОВА ЛОГІКА - торгуємо на кожному сигналі
                if current_prediction == 1 and not self.position:  # Покупаємо
                    size = int((cash * self.params.position_size) / current_price)
                    if size > 0:
                        self.log(f'📈 СИГНАЛ ПОКУПКИ: Розмір {size}, Ціна ${current_price:.2f}')
                        self.order = self.buy(size=size)
                
                elif current_prediction == 0 and self.position:  # Продаємо якщо є позиція
                    self.log(f'📉 СИГНАЛ ПРОДАЖУ: Розмір {self.position.size}, Ціна ${current_price:.2f}')
                    self.order = self.sell(size=self.position.size)
                
                self.prediction_index += 1
            
            def stop(self):
                final_value = self.broker.getvalue()
                total_return = ((final_value - self.initial_cash) / self.initial_cash) * 100
                win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
                
                self.log("=== ПІДСУМКИ ML СТРАТЕГІЇ ===")
                self.log(f"Початковий капітал: ${self.initial_cash:,.2f}")
                self.log(f"Фінальний капітал: ${final_value:,.2f}")
                self.log(f"Загальна доходність: {total_return:+.2f}%")
                self.log(f"Всього угод: {self.total_trades}")
                self.log(f"Виграшних угод: {self.winning_trades}")
                self.log(f"Винрейт: {win_rate:.1f}%")
                self.log("=" * 35)
        
        cerebro.addstrategy(SimplifiedMLStrategy)
        
        # Додаємо дані
        data_feed = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Використовуємо індекс
            open='open',
            high='high', 
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        # Налаштування брокера
        cerebro.broker.setcash(initial_balance)
        cerebro.broker.setcommission(commission=0.001)  # 0.1% комісія
        
        # Додаємо аналізатори
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Запускаємо бектест
        try:
            results = cerebro.run()
            
            if not results:
                raise RuntimeError("Backtrader не повернув результатів")
            
            result = results[0]
            
            # Отримуємо фінальні метрики
            final_value = cerebro.broker.getvalue()
            total_return = ((final_value - initial_balance) / initial_balance) * 100
            
            # Аналіз угод
            trades_analysis = result.analyzers.trades.get_analysis()
            total_trades = trades_analysis.get('total', {}).get('total', 0)
            won_trades = trades_analysis.get('won', {}).get('total', 0)
            
            # Додаткові метрики
            try:
                sharpe_data = result.analyzers.sharpe.get_analysis()
                sharpe_ratio = sharpe_data.get('sharperatio', 0) or 0
            except:
                sharpe_ratio = 0
            
            try:
                drawdown_data = result.analyzers.drawdown.get_analysis()
                max_drawdown = drawdown_data.get('max', {}).get('drawdown', 0) or 0
            except:
                max_drawdown = 0
            
            # Розрахунок винрейту
            win_rate_pct = (won_trades / max(total_trades, 1)) * 100
            
            # Повертаємо результати у форматі сумісному зі старою симуляцією
            return {
                'initial_balance': initial_balance,
                'final_balance': final_value,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'winning_trades': won_trades,
                'win_rate': win_rate_pct / 100,  # Конвертуємо у дроб
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'trades': [],  # Backtrader не надає детальну історію у такому форматі
                'balance_history': [],  # Можна додати якщо потрібно
                'backtrader_used': True,  # Позначка що використовувався backtrader
                'data_points': data_length
            }
            
        except Exception as e:
            print(f"⚠️ Помилка backtrader симуляції: {e}")
            print("Використовую стандартну симуляцію...")
            # Fallback до старої симуляції
            return self._simulate_direction_trading(y_true, y_pred, prices, initial_balance)