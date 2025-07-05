"""
Расширенные метрики для оценки торговых стратегий STAS_ML агентов.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradingMetrics:
    """Класс для хранения торговых метрик."""
    
    # Основные метрики доходности
    initial_balance: float
    final_balance: float
    total_return_pct: float
    total_return_usd: float
    
    # Метрики риска
    max_drawdown_pct: float
    max_drawdown_usd: float
    avg_drawdown_pct: float
    volatility: float
    
    # Торговые метрики
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit_per_trade: float
    profit_factor: float
    
    # Коэффициенты эффективности
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Временные метрики
    trading_days: int
    avg_daily_return: float
    best_day: float
    worst_day: float
    
    # Дополнительные метрики
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float  # Чистая прибыль / Максимальная просадка
    
    def to_dict(self) -> Dict:
        """Преобразовать в словарь."""
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'total_return_pct': self.total_return_pct,
            'total_return_usd': self.total_return_usd,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_drawdown_usd': self.max_drawdown_usd,
            'avg_drawdown_pct': self.avg_drawdown_pct,
            'volatility': self.volatility,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'trading_days': self.trading_days,
            'avg_daily_return': self.avg_daily_return,
            'best_day': self.best_day,
            'worst_day': self.worst_day,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'recovery_factor': self.recovery_factor
        }
    
    def print_summary(self):
        """Вывести краткую сводку метрик."""
        print("=" * 60)
        print("📊 ТОРГОВЫЕ МЕТРИКИ")
        print("=" * 60)
        print(f"💰 Начальный баланс:     ${self.initial_balance:,.2f}")
        print(f"💰 Итоговый баланс:      ${self.final_balance:,.2f}")
        print(f"📈 Общая доходность:     {self.total_return_pct:.2f}% (${self.total_return_usd:,.2f})")
        print(f"📉 Макс. просадка:       {self.max_drawdown_pct:.2f}% (${self.max_drawdown_usd:,.2f})")
        print(f"📊 Коэффициент Шарпа:    {self.sharpe_ratio:.3f}")
        print(f"🎯 Винрейт:             {self.win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
        print(f"💵 Прибыль на сделку:    ${self.avg_profit_per_trade:.2f}")
        print(f"🔄 Фактор прибыли:       {self.profit_factor:.2f}")
        print("=" * 60)


class MetricsCalculator:
    """Калькулятор торговых метрик."""
    
    @staticmethod
    def calculate_comprehensive_metrics(
        portfolio_history: List[float],
        trade_history: List[Dict],
        initial_balance: float,
        risk_free_rate: float = 0.02  # 2% годовых
    ) -> TradingMetrics:
        """
        Рассчитать комплексные торговые метрики.
        
        Args:
            portfolio_history: История стоимости портфеля
            trade_history: История сделок
            initial_balance: Начальный баланс
            risk_free_rate: Безрисковая ставка (для Sharpe ratio)
        """
        
        if not portfolio_history or len(portfolio_history) < 2:
            return MetricsCalculator._create_empty_metrics(initial_balance)
        
        # Основные метрики доходности
        final_balance = portfolio_history[-1]
        total_return_usd = final_balance - initial_balance
        total_return_pct = (total_return_usd / initial_balance) * 100
        
        # Метрики просадки
        drawdowns = MetricsCalculator._calculate_drawdowns(portfolio_history)
        max_drawdown_pct = max(drawdowns) * 100 if drawdowns else 0
        max_drawdown_usd = (max(drawdowns) * max(portfolio_history)) if drawdowns else 0
        avg_drawdown_pct = np.mean(drawdowns) * 100 if drawdowns else 0
        
        # Волатильность (дневная)
        returns = MetricsCalculator._calculate_returns(portfolio_history)
        volatility = np.std(returns) * 100 if len(returns) > 1 else 0
        
        # Торговые метрики
        trade_metrics = MetricsCalculator._calculate_trade_metrics(trade_history)
        
        # Коэффициенты эффективности
        sharpe_ratio = MetricsCalculator._calculate_sharpe_ratio(returns, risk_free_rate)
        calmar_ratio = MetricsCalculator._calculate_calmar_ratio(total_return_pct, max_drawdown_pct)
        sortino_ratio = MetricsCalculator._calculate_sortino_ratio(returns, risk_free_rate)
        
        # Временные метрики
        trading_days = len(portfolio_history)
        avg_daily_return = np.mean(returns) * 100 if returns else 0
        best_day = max(returns) * 100 if returns else 0
        worst_day = min(returns) * 100 if returns else 0
        
        # Дополнительные метрики
        consecutive_metrics = MetricsCalculator._calculate_consecutive_metrics(trade_history)
        recovery_factor = total_return_usd / max_drawdown_usd if max_drawdown_usd > 0 else 0
        
        return TradingMetrics(
            initial_balance=initial_balance,
            final_balance=final_balance,
            total_return_pct=total_return_pct,
            total_return_usd=total_return_usd,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_usd=max_drawdown_usd,
            avg_drawdown_pct=avg_drawdown_pct,
            volatility=volatility,
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate=trade_metrics['win_rate'],
            avg_profit_per_trade=trade_metrics['avg_profit_per_trade'],
            profit_factor=trade_metrics['profit_factor'],
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            trading_days=trading_days,
            avg_daily_return=avg_daily_return,
            best_day=best_day,
            worst_day=worst_day,
            max_consecutive_wins=consecutive_metrics['max_consecutive_wins'],
            max_consecutive_losses=consecutive_metrics['max_consecutive_losses'],
            recovery_factor=recovery_factor
        )
    
    @staticmethod
    def _create_empty_metrics(initial_balance: float) -> TradingMetrics:
        """Создать пустые метрики."""
        return TradingMetrics(
            initial_balance=initial_balance,
            final_balance=initial_balance,
            total_return_pct=0.0,
            total_return_usd=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_usd=0.0,
            avg_drawdown_pct=0.0,
            volatility=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_profit_per_trade=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            trading_days=0,
            avg_daily_return=0.0,
            best_day=0.0,
            worst_day=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            recovery_factor=0.0
        )
    
    @staticmethod
    def _calculate_drawdowns(portfolio_history: List[float]) -> List[float]:
        """Рассчитать просадки."""
        if not portfolio_history:
            return []
        
        drawdowns = []
        peak = portfolio_history[0]
        
        for value in portfolio_history:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0
            drawdowns.append(drawdown)
        
        return drawdowns
    
    @staticmethod
    def _calculate_returns(portfolio_history: List[float]) -> List[float]:
        """Рассчитать дневные доходности."""
        if len(portfolio_history) < 2:
            return []
        
        returns = []
        for i in range(1, len(portfolio_history)):
            if portfolio_history[i-1] > 0:
                ret = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                returns.append(ret)
        
        return returns
    
    @staticmethod
    def _calculate_trade_metrics(trade_history: List[Dict]) -> Dict:
        """Рассчитать метрики по сделкам."""
        if not trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'profit_factor': 0.0
            }
        
        # Фильтруем только сделки продажи (для расчета прибыли)
        sell_trades = [trade for trade in trade_history if trade.get('type') == 'sell']
        
        if not sell_trades:
            return {
                'total_trades': len(trade_history),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_profit_per_trade': 0.0,
                'profit_factor': 0.0
            }
        
        profits = [trade.get('profit', 0) for trade in sell_trades]
        winning_trades = sum(1 for profit in profits if profit > 0)
        losing_trades = sum(1 for profit in profits if profit < 0)
        
        win_rate = (winning_trades / len(sell_trades)) * 100 if sell_trades else 0
        avg_profit_per_trade = np.mean(profits) if profits else 0
        
        # Фактор прибыли = общая прибыль / общий убыток
        total_profit = sum(profit for profit in profits if profit > 0)
        total_loss = abs(sum(profit for profit in profits if profit < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        return {
            'total_trades': len(sell_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit_per_trade': avg_profit_per_trade,
            'profit_factor': profit_factor
        }
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
        """Рассчитать коэффициент Шарпа."""
        if not returns or len(returns) < 2:
            return 0.0
        
        # Приводим к дневной безрисковой ставке
        daily_risk_free = risk_free_rate / 252  # 252 торговых дня в году
        
        excess_returns = [ret - daily_risk_free for ret in returns]
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(252)  # Аннуализируем
    
    @staticmethod
    def _calculate_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """Рассчитать коэффициент Кальмара."""
        if max_drawdown <= 0:
            return 0.0
        return annual_return / max_drawdown
    
    @staticmethod
    def _calculate_sortino_ratio(returns: List[float], risk_free_rate: float) -> float:
        """Рассчитать коэффициент Сортино."""
        if not returns or len(returns) < 2:
            return 0.0
        
        daily_risk_free = risk_free_rate / 252
        excess_returns = [ret - daily_risk_free for ret in returns]
        
        # Считаем только отрицательные отклонения
        negative_returns = [ret for ret in excess_returns if ret < 0]
        
        if not negative_returns:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_deviation
        return sortino * np.sqrt(252)  # Аннуализируем
    
    @staticmethod
    def _calculate_consecutive_metrics(trade_history: List[Dict]) -> Dict:
        """Рассчитать метрики последовательных побед/поражений."""
        if not trade_history:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
        sell_trades = [trade for trade in trade_history if trade.get('type') == 'sell']
        
        if not sell_trades:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in sell_trades:
            profit = trade.get('profit', 0)
            
            if profit > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif profit < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }


def calculate_metrics(portfolio_history: List[float], trade_history: List[Dict], 
                     initial_balance: float) -> TradingMetrics:
    """
    Основная функция для расчета торговых метрик.
    
    Args:
        portfolio_history: История стоимости портфеля
        trade_history: История сделок
        initial_balance: Начальный баланс
        
    Returns:
        TradingMetrics: Объект с расчитанными метриками
    """
    return MetricsCalculator.calculate_comprehensive_metrics(
        portfolio_history, trade_history, initial_balance
    )


def compare_strategies(metrics_list: List[TradingMetrics], strategy_names: List[str]) -> pd.DataFrame:
    """
    Сравнить несколько стратегий.
    
    Args:
        metrics_list: Список метрик для каждой стратегии
        strategy_names: Названия стратегий
        
    Returns:
        DataFrame с сравнением стратегий
    """
    comparison_data = []
    
    for metrics, name in zip(metrics_list, strategy_names):
        comparison_data.append({
            'Strategy': name,
            'Total Return %': f"{metrics.total_return_pct:.2f}%",
            'Max Drawdown %': f"{metrics.max_drawdown_pct:.2f}%",
            'Sharpe Ratio': f"{metrics.sharpe_ratio:.3f}",
            'Win Rate %': f"{metrics.win_rate:.1f}%",
            'Profit Factor': f"{metrics.profit_factor:.2f}",
            'Recovery Factor': f"{metrics.recovery_factor:.2f}",
            'Total Trades': metrics.total_trades
        })
    
    return pd.DataFrame(comparison_data) 