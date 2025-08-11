"""
Интерактивная панель для визуализации и тестирования торговой системы.
Использует Streamlit для создания web-интерфейса.

Запуск:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Импорты из нашего пакета
try:
    from env import CryptoTradingEnv, ActionSpace, RewardType
    from data_loader import DataLoader
    from simulator import SlippageModel
    from metrics import MetricsCalculator, PerformanceMetrics
    from visualization import TradingVisualizer
except ImportError:
    from .env import CryptoTradingEnv, ActionSpace, RewardType
    from .data_loader import DataLoader
    from .simulator import SlippageModel
    from .metrics import MetricsCalculator
    from .visualization import TradingVisualizer


# Конфигурация страницы
st.set_page_config(
    page_title="Crypto Trading DRL Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class TradingDashboard:
    """Интерактивная панель управления торговой системой."""

    def __init__(self):
        """Инициализация dashboard."""
        self.initialize_session_state()

    @staticmethod
    def initialize_session_state():
        """Инициализация состояния сессии."""
        if 'env' not in st.session_state:
            st.session_state.env = None
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        if 'is_running' not in st.session_state:
            st.session_state.is_running = False
        if 'episode_data' not in st.session_state:
            st.session_state.episode_data = []
        if 'trades_log' not in st.session_state:
            st.session_state.trades_log = []
        if 'manual_mode' not in st.session_state:
            st.session_state.manual_mode = False

    def run(self):
        """Запустить dashboard."""
        # Заголовок
        st.markdown('<p class="main-header">📈 Crypto Trading DRL Dashboard</p>', unsafe_allow_html=True)

        # Sidebar - Настройки
        with st.sidebar:
            st.header("⚙️ Configuration")
            self.render_configuration_panel()

            st.divider()
            st.header("🎮 Controls")
            self.render_control_panel()

            st.divider()
            st.header("📊 Quick Stats")
            self.render_quick_stats()

        # Main area - вкладки
        tabs = st.tabs([
            "📊 Live Trading",
            "📈 Charts",
            "📝 Trade Log",
            "📉 Metrics",
            "🔧 Debug"
        ])

        with tabs[0]:
            self.render_live_trading_tab()

        with tabs[1]:
            self.render_charts_tab()

        with tabs[2]:
            self.render_trade_log_tab()

        with tabs[3]:
            self.render_metrics_tab()

        with tabs[4]:
            self.render_debug_tab()

    def render_configuration_panel(self):
        """Панель конфигурации окружения."""
        st.subheader("Environment Setup")

        # Выбор данных
        symbol = st.selectbox(
            "Trading Pair",
            ["BTCUSDT", "BTCUSDC", "ETHUSDT"],
            key="symbol"
        )

        timeframe = st.selectbox(
            "Timeframe",
            ["15m", "1h", "4h", "1d"],
            index=3,
            key="timeframe"
        )

        # Торговые параметры
        st.subheader("Trading Parameters")

        initial_balance = st.number_input(
            "Initial Balance ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
            key="initial_balance"
        )

        # Action space
        action_type = st.selectbox(
            "Action Space",
            ["Discrete (Hold/Buy/Sell)", "Continuous"],
            key="action_type_select"
        )

        # Reward type
        reward_type = st.selectbox(
            "Reward Function",
            ["PNL", "Log Return", "Sharpe", "Risk-Adjusted", "Sortino"],
            index=3,
            key="reward_type_select"
        )

        # Market simulation
        st.subheader("Market Simulation")

        col1, col2 = st.columns(2)
        with col1:
            maker_fee = st.number_input(
                "Maker Fee (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                key="maker_fee"
            ) / 100

        with col2:
            taker_fee = st.number_input(
                "Taker Fee (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="taker_fee"
            ) / 100

        slippage = st.slider(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="slippage"
        ) / 100

        # Кнопка инициализации
        if st.button("🚀 Initialize Environment", use_container_width=True):
            self.initialize_environment(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                action_type=action_type,
                reward_type=reward_type,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                slippage=slippage
            )

    def initialize_environment(
        self,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        action_type: str,
        reward_type: str,
        maker_fee: float,
        taker_fee: float,
        slippage: float
    ):
        """Инициализировать торговое окружение."""
        try:
            # Маппинг типов
            action_space_map = {
                "Discrete (Hold/Buy/Sell)": ActionSpace.DISCRETE,
                "Continuous": ActionSpace.CONTINUOUS
            }

            reward_type_map = {
                "PNL": RewardType.PNL,
                "Log Return": RewardType.LOG_RETURN,
                "Sharpe": RewardType.SHARPE,
                "Risk-Adjusted": RewardType.RISK_ADJUSTED,
                "Sortino": RewardType.SORTINO
            }

            # Создаем окружение
            env = CryptoTradingEnv(
                symbol=symbol,
                timeframe=timeframe,
                initial_balance=initial_balance,
                action_type=action_space_map[action_type],
                reward_type=reward_type_map[reward_type],
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                slippage_percentage=slippage,
                observation_window=50
            )

            st.session_state.env = env
            st.session_state.current_step = 0
            st.session_state.episode_data = []
            st.session_state.trades_log = []

            # Reset environment
            obs, info = env.reset()

            st.success(f"✅ Environment initialized! Loaded {len(env.data_loader)} candles.")
            st.info(f"Observation space: {env.observation_space.shape}")
            st.info(f"Action space: {env.action_space}")

        except Exception as e:
            st.error(f"❌ Error initializing environment: {str(e)}")

    def render_control_panel(self):
        """Панель управления симуляцией."""
        if st.session_state.env is None:
            st.warning("⚠️ Initialize environment first!")
            return

        # Режим работы
        mode = st.radio(
            "Mode",
            ["Manual Trading", "Auto Random", "Load Agent"],
            key="control_mode"
        )

        st.session_state.manual_mode = (mode == "Manual Trading")

        if mode == "Manual Trading":
            st.subheader("Manual Actions")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🟢 BUY", use_container_width=True):
                    self.execute_action(1)  # Buy action

            with col2:
                if st.button("⚪ HOLD", use_container_width=True):
                    self.execute_action(0)  # Hold action

            with col3:
                if st.button("🔴 SELL", use_container_width=True):
                    self.execute_action(2)  # Sell action

        elif mode == "Auto Random":
            col1, col2 = st.columns(2)

            with col1:
                if st.button("▶️ Start", use_container_width=True):
                    st.session_state.is_running = True

            with col2:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.session_state.is_running = False

            steps = st.slider("Steps per run", 1, 100, 10)

            if st.button("⏭️ Step Forward", use_container_width=True):
                for _ in range(steps):
                    if st.session_state.env is not None:
                        action = st.session_state.env.action_space.sample()
                        self.execute_action(action)

        # Reset button
        st.divider()
        if st.button("🔄 Reset Episode", use_container_width=True, type="secondary"):
            self.reset_episode()

    def execute_action(self, action: int):
        """Выполнить действие в окружении."""
        if st.session_state.env is None:
            return

        try:
            obs, reward, terminated, truncated, info = st.session_state.env.step(action)

            # Сохраняем данные
            step_data = {
                'step': st.session_state.current_step,
                'action': action,
                'reward': reward,
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance'],
                'crypto_held': info['crypto_held'],
                'price': info['current_price'],
                'trade_executed': info.get('trade_executed', False)
            }

            st.session_state.episode_data.append(step_data)

            if info.get('trade_executed'):
                st.session_state.trades_log.append(step_data)

            st.session_state.current_step += 1

            if terminated or truncated:
                st.warning("Episode finished!")
                st.session_state.is_running = False

        except Exception as e:
            st.error(f"Error executing action: {str(e)}")

    def reset_episode(self):
        """Сбросить эпизод."""
        if st.session_state.env is not None:
            st.session_state.env.reset()
            st.session_state.current_step = 0
            st.session_state.episode_data = []
            st.session_state.trades_log = []
            st.success("Episode reset!")

    def render_quick_stats(self):
        """Быстрые статистики."""
        if st.session_state.env is None or not st.session_state.episode_data:
            st.info("No data yet")
            return

        latest = st.session_state.episode_data[-1]
        initial = st.session_state.env.initial_balance

        portfolio_value = latest['portfolio_value']
        pnl = portfolio_value - initial
        pnl_pct = (pnl / initial) * 100

        st.metric("Portfolio Value", f"${portfolio_value:.2f}", f"{pnl_pct:+.2f}%")
        st.metric("Steps", st.session_state.current_step)
        st.metric("Trades", len(st.session_state.trades_log))

    def render_live_trading_tab(self):
        """Вкладка live trading."""
        st.header("Live Trading View")

        if st.session_state.env is None:
            st.info("👈 Initialize environment in the sidebar")
            return

        if not st.session_state.episode_data:
            st.info("Execute some actions to see data")
            return

        # Текущее состояние
        col1, col2, col3, col4 = st.columns(4)

        latest = st.session_state.episode_data[-1]
        initial = st.session_state.env.initial_balance

        with col1:
            st.metric(
                "Balance",
                f"${latest['balance']:.2f}",
                delta=f"{((latest['balance']/initial - 1) * 100):.2f}%"
            )

        with col2:
            st.metric(
                "Crypto Holdings",
                f"{latest['crypto_held']:.6f}",
                delta=f"${latest['crypto_held'] * latest['price']:.2f}"
            )

        with col3:
            pnl = latest['portfolio_value'] - initial
            st.metric(
                "P&L",
                f"${pnl:.2f}",
                delta=f"{(pnl/initial * 100):.2f}%"
            )

        with col4:
            st.metric(
                "Current Price",
                f"${latest['price']:.2f}"
            )

        # График equity curve в реальном времени
        st.subheader("Portfolio Value Over Time")

        equity_data = [d['portfolio_value'] for d in st.session_state.episode_data]
        steps = list(range(len(equity_data)))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=steps,
            y=equity_data,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ))

        fig.add_hline(y=initial, line_dash="dash", line_color="gray", annotation_text="Initial Balance")

        fig.update_layout(
            xaxis_title="Steps",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # График rewards
        st.subheader("Rewards Over Time")

        rewards = [d['reward'] for d in st.session_state.episode_data]

        fig_rewards = go.Figure()

        fig_rewards.add_trace(go.Scatter(
            x=steps,
            y=rewards,
            mode='lines',
            name='Reward',
            line=dict(color='blue', width=1)
        ))

        fig_rewards.add_hline(y=0, line_dash="dash", line_color="gray")

        fig_rewards.update_layout(
            xaxis_title="Steps",
            yaxis_title="Reward",
            hovermode='x unified',
            height=300
        )

        st.plotly_chart(fig_rewards, use_container_width=True)

    def render_charts_tab(self):
        """Вкладка с графиками."""
        st.header("Advanced Charts")

        if st.session_state.env is None:
            st.info("Initialize environment first")
            return

        # Candlestick chart
        st.subheader("Price Chart")

        try:
            data = st.session_state.env.data_loader.raw_data

            # Ограничиваем данные для производительности
            max_candles = st.slider("Max candles to display", 50, 500, 200)
            start_idx = max(0, st.session_state.current_step - max_candles)
            end_idx = st.session_state.current_step + 50

            data_slice = data.iloc[start_idx:end_idx]

            fig = go.Figure(data=[go.Candlestick(
                x=data_slice.index,
                open=data_slice['open'],
                high=data_slice['high'],
                low=data_slice['low'],
                close=data_slice['close'],
                name='OHLC'
            )])

            # Добавляем метки сделок
            if st.session_state.trades_log:
                buy_trades = [t for t in st.session_state.trades_log if t.get('action') == 1]
                sell_trades = [t for t in st.session_state.trades_log if t.get('action') == 2]

                if buy_trades:
                    buy_steps = [t['step'] for t in buy_trades]
                    buy_prices = [t['price'] for t in buy_trades]
                    fig.add_trace(go.Scatter(
                        x=buy_steps,
                        y=buy_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=15, color='green'),
                        name='Buy'
                    ))

                if sell_trades:
                    sell_steps = [t['step'] for t in sell_trades]
                    sell_prices = [t['price'] for t in sell_trades]
                    fig.add_trace(go.Scatter(
                        x=sell_steps,
                        y=sell_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=15, color='red'),
                        name='Sell'
                    ))

            # Вертикальная линия текущего шага
            if st.session_state.current_step > 0:
                fig.add_vline(
                    x=st.session_state.current_step,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Current"
                )

            fig.update_layout(
                xaxis_title="Step",
                yaxis_title="Price (USDT)",
                hovermode='x unified',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")

        # Drawdown chart
        if st.session_state.episode_data:
            st.subheader("Drawdown Analysis")

            equity_data = np.array([d['portfolio_value'] for d in st.session_state.episode_data])
            running_max = np.maximum.accumulate(equity_data)
            drawdown_pct = (equity_data - running_max) / running_max * 100

            fig_dd = go.Figure()

            fig_dd.add_trace(go.Scatter(
                x=list(range(len(drawdown_pct))),
                y=drawdown_pct,
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1),
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))

            fig_dd.update_layout(
                xaxis_title="Steps",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                height=300
            )

            st.plotly_chart(fig_dd, use_container_width=True)

    def render_trade_log_tab(self):
        """Вкладка с логом сделок."""
        st.header("Trade Log")

        if not st.session_state.trades_log:
            st.info("No trades executed yet")
            return

        # Создаем DataFrame
        trades_df = pd.DataFrame(st.session_state.trades_log)

        # Добавляем человеко-читаемые action names
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        trades_df['Action Name'] = trades_df['action'].map(action_names)

        # Форматируем колонки
        display_df = trades_df[[
            'step', 'Action Name', 'price', 'balance',
            'crypto_held', 'portfolio_value', 'reward'
        ]].copy()

        display_df.columns = [
            'Step', 'Action', 'Price', 'Balance',
            'Crypto', 'Portfolio', 'Reward'
        ]

        # Форматирование
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Balance'] = display_df['Balance'].apply(lambda x: f"${x:.2f}")
        display_df['Crypto'] = display_df['Crypto'].apply(lambda x: f"{x:.6f}")
        display_df['Portfolio'] = display_df['Portfolio'].apply(lambda x: f"${x:.2f}")
        display_df['Reward'] = display_df['Reward'].apply(lambda x: f"{x:.4f}")

        st.dataframe(display_df, use_container_width=True, height=400)

        # Кнопка экспорта
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log (CSV)",
            data=csv,
            file_name=f"trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    def render_metrics_tab(self):
        """Вкладка с метриками."""
        st.header("Performance Metrics")

        if st.session_state.env is None or not st.session_state.episode_data:
            st.info("No data to calculate metrics")
            return

        # Рассчитываем метрики
        equity_curve = [d['portfolio_value'] for d in st.session_state.episode_data]
        metrics = st.session_state.env.get_metrics()

        # Основные метрики
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Returns")
            st.metric("Total Return", f"{metrics.total_return_pct:.2f}%")
            st.metric("Annualized Return", f"{metrics.annualized_return:.2f}%")

        with col2:
            st.markdown("### Risk")
            st.metric("Volatility", f"{metrics.annualized_volatility:.2f}%")
            st.metric("Max Drawdown", f"{metrics.max_drawdown_pct:.2f}%")

        with col3:
            st.markdown("### Risk-Adjusted")
            st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
            st.metric("Sortino Ratio", f"{metrics.sortino_ratio:.2f}")

        st.divider()

        # Торговые метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades", metrics.total_trades)

        with col2:
            st.metric("Win Rate", f"{metrics.win_rate:.1f}%")

        with col3:
            st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

        with col4:
            st.metric("Avg Trade", f"${metrics.avg_trade_return:.2f}")

        # Детальная таблица
        st.subheader("Detailed Metrics")

        metrics_data = {
            "Metric": [
                "Total Return", "Total Return %", "Annualized Return",
                "Volatility", "Annualized Volatility", "Downside Volatility",
                "Max Drawdown", "Max Drawdown %", "Average Drawdown",
                "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                "Total Trades", "Winning Trades", "Losing Trades",
                "Win Rate", "Profit Factor", "Avg Win", "Avg Loss",
                "Largest Win", "Largest Loss"
            ],
            "Value": [
                f"${metrics.total_return:.2f}",
                f"{metrics.total_return_pct:.2f}%",
                f"{metrics.annualized_return:.2f}%",
                f"{metrics.volatility:.2f}%",
                f"{metrics.annualized_volatility:.2f}%",
                f"{metrics.downside_volatility:.2f}%",
                f"${metrics.max_drawdown:.2f}",
                f"{metrics.max_drawdown_pct:.2f}%",
                f"${metrics.avg_drawdown:.2f}",
                f"{metrics.sharpe_ratio:.2f}",
                f"{metrics.sortino_ratio:.2f}",
                f"{metrics.calmar_ratio:.2f}",
                str(metrics.total_trades),
                str(metrics.winning_trades),
                str(metrics.losing_trades),
                f"{metrics.win_rate:.2f}%",
                f"{metrics.profit_factor:.2f}",
                f"${metrics.avg_win:.2f}",
                f"${metrics.avg_loss:.2f}",
                f"${metrics.largest_win:.2f}",
                f"${metrics.largest_loss:.2f}"
            ]
        }

        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, height=400)

    def render_debug_tab(self):
        """Вкладка отладки."""
        st.header("Debug Information")

        if st.session_state.env is None:
            st.info("No environment loaded")
            return

        # Observation space
        st.subheader("Observation Space")
        st.code(f"Shape: {st.session_state.env.observation_space.shape}")
        st.code(f"Type: {st.session_state.env.observation_space.dtype}")

        # Action space
        st.subheader("Action Space")
        st.code(f"Type: {st.session_state.env.action_space}")

        # Current observation
        if st.session_state.episode_data:
            st.subheader("Latest Step Info")

            latest = st.session_state.episode_data[-1]

            st.json(latest)

        # Environment state
        st.subheader("Environment State")

        env_state = {
            "Current Step": st.session_state.env.current_step,
            "Data Length": len(st.session_state.env.data_loader),
            "Initial Balance": st.session_state.env.initial_balance,
            "Current Balance": st.session_state.env.balance,
            "Crypto Held": st.session_state.env.crypto_held,
            "Total Trades": st.session_state.env.total_trades
        }

        st.json(env_state)

        # Session state
        st.subheader("Session State")
        st.write({
            "Current Step": st.session_state.current_step,
            "Is Running": st.session_state.is_running,
            "Manual Mode": st.session_state.manual_mode,
            "Episode Data Points": len(st.session_state.episode_data),
            "Trades Log Size": len(st.session_state.trades_log)
        })


def main():
    """Main entry point."""
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
