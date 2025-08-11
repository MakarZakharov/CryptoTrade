"""
Модуль расчета торговых метрик и показателей производительности.
Включает Sharpe, Sortino, Calmar, drawdown, win rate и другие метрики.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class TradeMetrics:
    """Метрики по отдельной сделке."""
    entry_time: float
    exit_time: Optional[float] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    quantity: float = 0.0
    side: str = "long"  # "long" or "short"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    holding_time: float = 0.0
    is_winner: bool = False


@dataclass
class PerformanceMetrics:
    """Комплексные метрики производительности."""
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    annualized_volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_time: float = 0.0

    # Exposure
    total_commission: float = 0.0
    total_slippage: float = 0.0
    avg_turnover: float = 0.0

    # Additional
    trades: List[TradeMetrics] = field(default_factory=list)


class MetricsCalculator:
    """
    Калькулятор торговых метрик.

    Рассчитывает все ключевые показатели производительности торговой системы.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Инициализация калькулятора метрик.

        Args:
            risk_free_rate: Безрисковая ставка (годовая)
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        equity_curve: List[float],
        trades: Optional[List[TradeMetrics]] = None,
        timestamps: Optional[List[float]] = None,
        initial_balance: float = 10000.0
    ) -> PerformanceMetrics:
        """
        Рассчитать все метрики производительности.

        Args:
            equity_curve: Кривая капитала (баланс на каждом шаге)
            trades: Список сделок
            timestamps: Временные метки для каждого шага
            initial_balance: Начальный баланс

        Returns:
            PerformanceMetrics с рассчитанными метриками
        """
        if not equity_curve or len(equity_curve) < 2:
            return PerformanceMetrics()

        equity_array = np.array(equity_curve)
        returns = self._calculate_returns(equity_array)

        metrics = PerformanceMetrics()

        # Basic returns
        metrics.total_return = equity_array[-1] - initial_balance
        metrics.total_return_pct = (equity_array[-1] / initial_balance - 1) * 100

        # Annualized return
        if timestamps is not None and len(timestamps) > 1:
            time_period_years = (timestamps[-1] - timestamps[0]) / (365 * 24 * 3600)
            if time_period_years > 0:
                metrics.annualized_return = (
                    (equity_array[-1] / initial_balance) ** (1 / time_period_years) - 1
                ) * 100
        else:
            # Предполагаем дневные данные
            periods_per_year = 252  # Торговых дней
            time_period_years = len(equity_curve) / periods_per_year
            if time_period_years > 0:
                metrics.annualized_return = (
                    (equity_array[-1] / initial_balance) ** (1 / time_period_years) - 1
                ) * 100

        # Volatility
        if len(returns) > 1:
            metrics.volatility = np.std(returns) * 100
            metrics.annualized_volatility = metrics.volatility * np.sqrt(252)

            # Downside volatility (только негативные возвраты)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                metrics.downside_volatility = np.std(downside_returns) * np.sqrt(252) * 100

        # Drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(equity_array)
        metrics.max_drawdown = dd_metrics['max_drawdown']
        metrics.max_drawdown_pct = dd_metrics['max_drawdown_pct']
        metrics.avg_drawdown = dd_metrics['avg_drawdown']
        metrics.drawdown_duration = dd_metrics['max_duration']

        # Risk-adjusted returns
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)

        if metrics.max_drawdown_pct != 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown_pct)

        metrics.omega_ratio = self._calculate_omega_ratio(returns)

        # Trade statistics
        if trades is not None and len(trades) > 0:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.winning_trades = trade_metrics['winning_trades']
            metrics.losing_trades = trade_metrics['losing_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.avg_trade_return = trade_metrics['avg_trade_return']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.largest_win = trade_metrics['largest_win']
            metrics.largest_loss = trade_metrics['largest_loss']
            metrics.avg_holding_time = trade_metrics['avg_holding_time']
            metrics.total_commission = trade_metrics['total_commission']
            metrics.trades = trades

        return metrics

    def _calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Рассчитать returns из кривой капитала."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns

    def _calculate_drawdown_metrics(self, equity_curve: np.ndarray) -> Dict[str, Any]:
        """
        Рассчитать метрики просадки.

        Returns:
            Словарь с метриками drawdown
        """
        # Running maximum
        running_max = np.maximum.accumulate(equity_curve)

        # Drawdown в абсолютных величинах
        drawdown = equity_curve - running_max

        # Drawdown в процентах
        drawdown_pct = (drawdown / running_max) * 100

        # Maximum drawdown
        max_dd = np.min(drawdown)
        max_dd_pct = np.min(drawdown_pct)

        # Average drawdown
        avg_dd = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0

        # Drawdown duration
        is_drawdown = drawdown < 0
        drawdown_periods = self._get_consecutive_periods(is_drawdown)
        max_duration = max(drawdown_periods) if drawdown_periods else 0

        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'avg_drawdown': avg_dd,
            'max_duration': max_duration,
            'drawdown_curve': drawdown
        }

    def _get_consecutive_periods(self, mask: np.ndarray) -> List[int]:
        """Получить длины последовательных периодов True в маске."""
        periods = []
        current_period = 0

        for value in mask:
            if value:
                current_period += 1
            else:
                if current_period > 0:
                    periods.append(current_period)
                current_period = 0

        if current_period > 0:
            periods.append(current_period)

        return periods

    def _calculate_sharpe_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Рассчитать Sharpe Ratio.

        Args:
            returns: Массив returns
            periods_per_year: Количество периодов в году

        Returns:
            Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0

        # Средний return
        mean_return = np.mean(returns)

        # Стандартное отклонение
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Дневная безрисковая ставка
        daily_rf = self.risk_free_rate / periods_per_year

        # Sharpe ratio
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(periods_per_year)

        return sharpe

    def _calculate_sortino_ratio(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """
        Рассчитать Sortino Ratio (использует только downside volatility).

        Args:
            returns: Массив returns
            periods_per_year: Количество периодов в году

        Returns:
            Sortino Ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns)

        # Downside deviation
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns, ddof=1)

        if downside_std == 0:
            return 0.0

        daily_rf = self.risk_free_rate / periods_per_year

        sortino = (mean_return - daily_rf) / downside_std * np.sqrt(periods_per_year)

        return sortino

    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Рассчитать Omega Ratio.

        Args:
            returns: Массив returns
            threshold: Пороговое значение return

        Returns:
            Omega Ratio
        """
        if len(returns) == 0:
            return 0.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        if len(losses) == 0 or np.sum(losses) == 0:
            return float('inf') if len(gains) > 0 else 0.0

        omega = np.sum(gains) / np.sum(losses)

        return omega

    def _calculate_trade_metrics(self, trades: List[TradeMetrics]) -> Dict[str, Any]:
        """
        Рассчитать метрики по сделкам.

        Args:
            trades: Список сделок

        Returns:
            Словарь с метриками сделок
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'avg_holding_time': 0.0,
                'total_commission': 0.0
            }

        # Разделяем на выигрышные и проигрышные
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        total_trades = len(trades)
        winning_count = len(winning_trades)
        losing_count = len(losing_trades)

        # Win rate
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0.0

        # PnL
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))

        # Profit factor
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        # Average trade
        avg_trade = np.mean([t.pnl for t in trades])

        # Average win/loss
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

        # Largest win/loss
        largest_win = max([t.pnl for t in trades]) if trades else 0.0
        largest_loss = min([t.pnl for t in trades]) if trades else 0.0

        # Average holding time
        avg_holding = np.mean([t.holding_time for t in trades if t.holding_time > 0])

        # Total commission
        total_commission = sum([t.commission for t in trades])

        return {
            'total_trades': total_trades,
            'winning_trades': winning_count,
            'losing_trades': losing_count,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_holding_time': avg_holding,
            'total_commission': total_commission
        }

    def calculate_rolling_sharpe(
        self,
        equity_curve: List[float],
        window: int = 30
    ) -> np.ndarray:
        """
        Рассчитать rolling Sharpe ratio.

        Args:
            equity_curve: Кривая капитала
            window: Размер окна

        Returns:
            Массив rolling Sharpe ratios
        """
        if len(equity_curve) < window + 1:
            return np.array([])

        equity_array = np.array(equity_curve)
        returns = self._calculate_returns(equity_array)

        rolling_sharpe = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            sharpe = self._calculate_sharpe_ratio(window_returns)
            rolling_sharpe.append(sharpe)

        return np.array(rolling_sharpe)

    def compare_with_baseline(
        self,
        strategy_metrics: PerformanceMetrics,
        baseline_equity: List[float],
        initial_balance: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Сравнить метрики стратегии с baseline (buy & hold).

        Args:
            strategy_metrics: Метрики стратегии
            baseline_equity: Кривая капитала baseline
            initial_balance: Начальный баланс

        Returns:
            Словарь с сравнением
        """
        baseline_metrics = self.calculate_metrics(
            equity_curve=baseline_equity,
            initial_balance=initial_balance
        )

        return {
            'strategy': {
                'return_pct': strategy_metrics.total_return_pct,
                'sharpe': strategy_metrics.sharpe_ratio,
                'max_dd_pct': strategy_metrics.max_drawdown_pct,
                'volatility': strategy_metrics.annualized_volatility
            },
            'baseline': {
                'return_pct': baseline_metrics.total_return_pct,
                'sharpe': baseline_metrics.sharpe_ratio,
                'max_dd_pct': baseline_metrics.max_drawdown_pct,
                'volatility': baseline_metrics.annualized_volatility
            },
            'improvement': {
                'return_pct': strategy_metrics.total_return_pct - baseline_metrics.total_return_pct,
                'sharpe': strategy_metrics.sharpe_ratio - baseline_metrics.sharpe_ratio,
                'max_dd_pct': strategy_metrics.max_drawdown_pct - baseline_metrics.max_drawdown_pct,
                'volatility': strategy_metrics.annualized_volatility - baseline_metrics.annualized_volatility
            }
        }


if __name__ == "__main__":
    # Тестирование калькулятора метрик
    print("=== Metrics Calculator Test ===\n")

    # Создаем симулированную кривую капитала
    np.random.seed(42)
    initial_balance = 10000.0

    # Симуляция equity curve с трендом вверх и волатильностью
    returns = np.random.normal(0.001, 0.02, 250)  # 250 дней
    equity_curve = [initial_balance]

    for ret in returns:
        equity_curve.append(equity_curve[-1] * (1 + ret))

    # Создаем несколько сделок
    trades = [
        TradeMetrics(
            entry_time=1.0,
            exit_time=2.0,
            entry_price=100.0,
            exit_price=105.0,
            quantity=1.0,
            side="long",
            pnl=5.0,
            pnl_pct=5.0,
            commission=0.1,
            holding_time=1.0,
            is_winner=True
        ),
        TradeMetrics(
            entry_time=3.0,
            exit_time=4.0,
            entry_price=105.0,
            exit_price=103.0,
            quantity=1.0,
            side="long",
            pnl=-2.0,
            pnl_pct=-1.9,
            commission=0.1,
            holding_time=1.0,
            is_winner=False
        ),
    ]

    # Рассчитываем метрики
    calculator = MetricsCalculator(risk_free_rate=0.02)
    metrics = calculator.calculate_metrics(
        equity_curve=equity_curve,
        trades=trades,
        initial_balance=initial_balance
    )

    # Выводим результаты
    print("Performance Metrics:")
    print(f"  Total Return: ${metrics.total_return:.2f} ({metrics.total_return_pct:.2f}%)")
    print(f"  Annualized Return: {metrics.annualized_return:.2f}%")
    print(f"  Volatility: {metrics.volatility:.2f}%")
    print(f"  Annualized Volatility: {metrics.annualized_volatility:.2f}%")
    print(f"  Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {metrics.total_trades}")
    print(f"  Win Rate: {metrics.win_rate:.2f}%")
    print(f"  Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  Avg Trade: ${metrics.avg_trade_return:.2f}")
