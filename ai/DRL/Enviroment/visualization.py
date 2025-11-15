"""
Модуль визуализации для торгового окружения.
Создает графики: candlestick, equity curve, drawdown, indicators, trades.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings

from .metrics import PerformanceMetrics, TradeMetrics


class TradingVisualizer:
    """
    Визуализатор торговых данных.

    Создает интерактивные и статические графики для анализа.
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 12), style: str = 'seaborn-v0_8-darkgrid'):
        """
        Инициализация визуализатора.

        Args:
            figsize: Размер фигуры matplotlib
            style: Стиль matplotlib
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

    def plot_full_analysis(
        self,
        data: pd.DataFrame,
        equity_curve: List[float],
        trades: List[TradeMetrics],
        metrics: PerformanceMetrics,
        symbol: str = "BTCUSDT",
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Создать полный анализ с несколькими графиками.

        Args:
            data: DataFrame с OHLCV данными
            equity_curve: Кривая капитала
            trades: Список сделок
            metrics: Метрики производительности
            symbol: Символ торговой пары
            save_path: Путь для сохранения
            show: Показать ли график
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Candlestick chart с ордерами
        ax1 = fig.add_subplot(gs[0:2, :])
        self._plot_candlestick_with_trades(ax1, data, trades, symbol)

        # 2. Equity curve
        ax2 = fig.add_subplot(gs[2, 0])
        self._plot_equity_curve(ax2, equity_curve)

        # 3. Drawdown
        ax3 = fig.add_subplot(gs[2, 1])
        self._plot_drawdown(ax3, equity_curve)

        # 4. Returns distribution
        ax4 = fig.add_subplot(gs[3, 0])
        self._plot_returns_distribution(ax4, equity_curve)

        # 5. Metrics summary
        ax5 = fig.add_subplot(gs[3, 1])
        self._plot_metrics_summary(ax5, metrics)

        plt.suptitle(f'{symbol} Trading Analysis', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _plot_candlestick_with_trades(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        trades: List[TradeMetrics],
        symbol: str
    ):
        """Построить candlestick график с метками сделок."""
        # Упрощенный candlestick (линия close + volume)
        if 'timestamp' in data.columns:
            x = pd.to_datetime(data['timestamp'])
        else:
            x = np.arange(len(data))

        # Цена закрытия
        ax.plot(x, data['close'], linewidth=1, label='Close Price', color='blue', alpha=0.7)

        # High/Low shadow
        for i in range(0, len(data), max(1, len(data) // 100)):  # Показываем каждый N-ый для производительности
            ax.plot([x.iloc[i], x.iloc[i]], [data['low'].iloc[i], data['high'].iloc[i]],
                   color='gray', linewidth=0.5, alpha=0.3)

        # Метки сделок
        buy_trades = [t for t in trades if t.side == 'long' and t.entry_time is not None]
        sell_trades = [t for t in trades if t.exit_time is not None]

        if buy_trades:
            buy_x = [x.iloc[int(t.entry_time)] if int(t.entry_time) < len(x) else x.iloc[-1] for t in buy_trades]
            buy_y = [t.entry_price for t in buy_trades]
            ax.scatter(buy_x, buy_y, marker='^', color='green', s=100, label='Buy', zorder=5)

        if sell_trades:
            sell_x = [x.iloc[int(t.exit_time)] if int(t.exit_time) < len(x) else x.iloc[-1] for t in sell_trades]
            sell_y = [t.exit_price for t in sell_trades]
            ax.scatter(sell_x, sell_y, marker='v', color='red', s=100, label='Sell', zorder=5)

        ax.set_title(f'{symbol} Price Chart with Trades', fontweight='bold')
        ax.set_ylabel('Price (USDT)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Форматирование оси X
        if 'timestamp' in data.columns:
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    def _plot_equity_curve(self, ax: plt.Axes, equity_curve: List[float]):
        """Построить кривую капитала."""
        ax.plot(equity_curve, linewidth=2, color='green', label='Portfolio Value')
        ax.axhline(y=equity_curve[0], color='gray', linestyle='--', alpha=0.5, label='Initial Balance')

        # Fill между начальным балансом и equity
        ax.fill_between(range(len(equity_curve)), equity_curve[0], equity_curve,
                        where=np.array(equity_curve) >= equity_curve[0],
                        color='green', alpha=0.2, label='Profit')
        ax.fill_between(range(len(equity_curve)), equity_curve[0], equity_curve,
                        where=np.array(equity_curve) < equity_curve[0],
                        color='red', alpha=0.2, label='Loss')

        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_drawdown(self, ax: plt.Axes, equity_curve: List[float]):
        """Построить график просадки."""
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown_pct = (equity_array - running_max) / running_max * 100

        ax.fill_between(range(len(drawdown_pct)), 0, drawdown_pct,
                       color='red', alpha=0.3)
        ax.plot(drawdown_pct, color='red', linewidth=1.5)

        ax.set_title('Drawdown', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # Отметить максимальную просадку
        max_dd_idx = np.argmin(drawdown_pct)
        ax.scatter([max_dd_idx], [drawdown_pct[max_dd_idx]],
                  color='darkred', s=100, zorder=5, label=f'Max DD: {drawdown_pct[max_dd_idx]:.2f}%')
        ax.legend(loc='best')

    def _plot_returns_distribution(self, ax: plt.Axes, equity_curve: List[float]):
        """Построить распределение returns."""
        if len(equity_curve) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return

        returns = np.diff(equity_curve) / equity_curve[:-1] * 100  # В процентах

        ax.hist(returns, bins=50, color='blue', alpha=0.6, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        ax.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(returns):.3f}%')

        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_metrics_summary(self, ax: plt.Axes, metrics: PerformanceMetrics):
        """Отобразить сводку метрик."""
        ax.axis('off')

        metrics_text = f"""
PERFORMANCE METRICS

Returns:
  Total Return: {metrics.total_return_pct:.2f}%
  Annualized: {metrics.annualized_return:.2f}%

Risk:
  Volatility: {metrics.annualized_volatility:.2f}%
  Max Drawdown: {metrics.max_drawdown_pct:.2f}%

Risk-Adjusted:
  Sharpe Ratio: {metrics.sharpe_ratio:.2f}
  Sortino Ratio: {metrics.sortino_ratio:.2f}
  Calmar Ratio: {metrics.calmar_ratio:.2f}

Trading:
  Total Trades: {metrics.total_trades}
  Win Rate: {metrics.win_rate:.2f}%
  Profit Factor: {metrics.profit_factor:.2f}
  Avg Trade: ${metrics.avg_trade_return:.2f}
        """

        ax.text(0.1, 0.9, metrics_text.strip(), transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    def create_interactive_plotly(
        self,
        data: pd.DataFrame,
        equity_curve: List[float],
        trades: List[TradeMetrics],
        metrics: PerformanceMetrics,
        symbol: str = "BTCUSDT",
        save_path: Optional[str] = None
    ):
        """
        Создать интерактивный график с Plotly.

        Args:
            data: DataFrame с OHLCV данными
            equity_curve: Кривая капитала
            trades: Список сделок
            metrics: Метрики производительности
            symbol: Символ
            save_path: Путь для сохранения HTML
        """
        # Создаем subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'{symbol} Candlestick Chart',
                'Equity Curve',
                'Volume',
                'Drawdown',
                'Returns Distribution',
                'Metrics Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Candlestick
        if 'timestamp' in data.columns:
            x = pd.to_datetime(data['timestamp'])
        else:
            x = np.arange(len(data))

        fig.add_trace(
            go.Candlestick(
                x=x,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Buy/Sell markers
        buy_trades = [t for t in trades if t.side == 'long' and t.entry_time is not None]
        sell_trades = [t for t in trades if t.exit_time is not None]

        if buy_trades:
            buy_x = [x.iloc[int(t.entry_time)] if int(t.entry_time) < len(x) else x.iloc[-1] for t in buy_trades]
            buy_y = [t.entry_price for t in buy_trades]
            fig.add_trace(
                go.Scatter(
                    x=buy_x, y=buy_y,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Buy'
                ),
                row=1, col=1
            )

        if sell_trades:
            sell_x = [x.iloc[int(t.exit_time)] if int(t.exit_time) < len(x) else x.iloc[-1] for t in sell_trades]
            sell_y = [t.exit_price for t in sell_trades]
            fig.add_trace(
                go.Scatter(
                    x=sell_x, y=sell_y,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Sell'
                ),
                row=1, col=1
            )

        # 2. Equity curve
        fig.add_trace(
            go.Scatter(
                y=equity_curve,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )

        # 3. Volume
        colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green'
                 for i in range(len(data))]

        fig.add_trace(
            go.Bar(x=x, y=data['volume'], name='Volume', marker_color=colors),
            row=2, col=1
        )

        # 4. Drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown_pct = (equity_array - running_max) / running_max * 100

        fig.add_trace(
            go.Scatter(
                y=drawdown_pct,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=2, col=2
        )

        # 5. Returns distribution
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1] * 100

            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=3, col=1
            )

        # 6. Metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f'{metrics.total_return_pct:.2f}%'],
            ['Annualized Return', f'{metrics.annualized_return:.2f}%'],
            ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
            ['Max Drawdown', f'{metrics.max_drawdown_pct:.2f}%'],
            ['Win Rate', f'{metrics.win_rate:.2f}%'],
            ['Total Trades', str(metrics.total_trades)],
            ['Profit Factor', f'{metrics.profit_factor:.2f}']
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*metrics_data[1:])),
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=3, col=2
        )

        # Layout
        fig.update_layout(
            title=f'{symbol} Trading Analysis',
            showlegend=True,
            height=1200,
            template='plotly_white'
        )

        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Steps", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Steps", row=2, col=2)
        fig.update_xaxes(title_text="Return (%)", row=3, col=1)

        fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=2)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)

        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive plot to {save_path}")
        else:
            fig.show()

        return fig

    def plot_rolling_metrics(
        self,
        equity_curve: List[float],
        window: int = 30,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Построить rolling метрики.

        Args:
            equity_curve: Кривая капитала
            window: Размер окна
            save_path: Путь для сохранения
            show: Показать график
        """
        if len(equity_curve) < window + 1:
            print("Insufficient data for rolling metrics")
            return

        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        # Rolling Sharpe
        rolling_sharpe = []
        rolling_volatility = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            sharpe = np.mean(window_returns) / (np.std(window_returns) + 1e-6) * np.sqrt(252)
            vol = np.std(window_returns) * np.sqrt(252) * 100
            rolling_sharpe.append(sharpe)
            rolling_volatility.append(vol)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Rolling Sharpe
        axes[0].plot(rolling_sharpe, linewidth=2, color='blue')
        axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'Rolling Sharpe Ratio (window={window})', fontweight='bold')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].grid(True, alpha=0.3)

        # Rolling Volatility
        axes[1].plot(rolling_volatility, linewidth=2, color='orange')
        axes[1].set_title(f'Rolling Volatility (window={window})', fontweight='bold')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Annualized Volatility (%)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # Тестирование визуализатора
    print("=== Trading Visualizer Test ===\n")

    # Создаем тестовые данные
    np.random.seed(42)
    n_steps = 250

    # OHLCV данные
    close_prices = 50000 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_steps))
    data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_steps, freq='D'),
        'open': close_prices * (1 + np.random.uniform(-0.01, 0.01, n_steps)),
        'high': close_prices * (1 + np.random.uniform(0, 0.02, n_steps)),
        'low': close_prices * (1 - np.random.uniform(0, 0.02, n_steps)),
        'close': close_prices,
        'volume': np.random.uniform(1000, 10000, n_steps)
    })

    # Equity curve
    initial_balance = 10000
    equity_curve = [initial_balance]
    for i in range(n_steps - 1):
        equity_curve.append(equity_curve[-1] * (1 + np.random.normal(0.002, 0.015)))

    # Тестовые сделки
    from .metrics import TradeMetrics
    trades = [
        TradeMetrics(
            entry_time=10,
            exit_time=20,
            entry_price=close_prices[10],
            exit_price=close_prices[20],
            quantity=0.1,
            side='long',
            pnl=(close_prices[20] - close_prices[10]) * 0.1,
            commission=5.0,
            holding_time=10,
            is_winner=True
        ),
        TradeMetrics(
            entry_time=50,
            exit_time=60,
            entry_price=close_prices[50],
            exit_price=close_prices[60],
            quantity=0.1,
            side='long',
            pnl=(close_prices[60] - close_prices[50]) * 0.1,
            commission=5.0,
            holding_time=10,
            is_winner=False
        )
    ]

    # Метрики (простые)
    from .metrics import PerformanceMetrics
    metrics = PerformanceMetrics(
        total_return_pct=15.5,
        annualized_return=25.3,
        sharpe_ratio=1.8,
        max_drawdown_pct=-12.5,
        win_rate=55.0,
        total_trades=10,
        profit_factor=1.5
    )

    # Создаем визуализатор
    viz = TradingVisualizer()

    # Статические графики
    print("Creating static plot...")
    viz.plot_full_analysis(
        data=data,
        equity_curve=equity_curve,
        trades=trades,
        metrics=metrics,
        symbol="BTCUSDT",
        save_path="test_analysis.png",
        show=False
    )

    print("Test completed!")
