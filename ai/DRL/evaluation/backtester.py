"""
Система бэктестинга для DRL торговых агентов.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Конфигурация для бэктестинга."""
    initial_capital: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    benchmark: str = 'buy_and_hold'  # 'buy_and_hold', 'random', 'sma_crossover'
    risk_free_rate: float = 0.02  # 2% годовых


class FinancialMetrics:
    """Класс для расчета финансовых метрик."""
    
    @staticmethod
    def total_return(portfolio_values: pd.Series) -> float:
        """Общая доходность."""
        return (portfolio_values.iloc[-1] - portfolio_values.iloc[0]) / portfolio_values.iloc[0]
    
    @staticmethod
    def annualized_return(portfolio_values: pd.Series, trading_days: int = 252) -> float:
        """Годовая доходность."""
        total_ret = FinancialMetrics.total_return(portfolio_values)
        days = len(portfolio_values)
        return (1 + total_ret) ** (trading_days / days) - 1
    
    @staticmethod
    def volatility(returns: pd.Series, trading_days: int = 252) -> float:
        """Волатильность (годовая)."""
        return returns.std() * np.sqrt(trading_days)
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """Коэффициент Шарпа."""
        excess_returns = returns - risk_free_rate / trading_days
        return excess_returns.mean() / returns.std() * np.sqrt(trading_days)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """Коэффициент Сортино."""
        excess_returns = returns - risk_free_rate / trading_days
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_deviation = downside_returns.std()
        return excess_returns.mean() / downside_deviation * np.sqrt(trading_days)
    
    @staticmethod
    def max_drawdown(portfolio_values: pd.Series) -> Tuple[float, int, int]:
        """
        Максимальная просадка.
        
        Returns:
            (max_drawdown, start_idx, end_idx)
        """
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        max_dd = drawdown.min()
        end_idx = drawdown.idxmin()
        start_idx = portfolio_values[:end_idx].idxmax()
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calmar_ratio(portfolio_values: pd.Series, trading_days: int = 252) -> float:
        """Коэффициент Калмара."""
        annual_return = FinancialMetrics.annualized_return(portfolio_values, trading_days)
        max_dd, _, _ = FinancialMetrics.max_drawdown(portfolio_values)
        
        if max_dd == 0:
            return np.inf
        return annual_return / abs(max_dd)
    
    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """Процент прибыльных сделок."""
        if not trades:
            return 0.0
        
        profitable_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return profitable_trades / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """Фактор прибыли (отношение прибылей к убыткам)."""
        if not trades:
            return 0.0
        
        profits = sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0)
        losses = abs(sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0))
        
        if losses == 0:
            return np.inf if profits > 0 else 0.0
        return profits / losses


class Backtester:
    """Основной класс для бэктестинга DRL агентов."""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.logger = self._setup_logger()
        self.results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger('Backtester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def run_backtest(
        self, 
        agent, 
        test_data: pd.DataFrame, 
        env_config: Dict = None
    ) -> Dict[str, Any]:
        """
        Запуск бэктестинга агента.
        
        Args:
            agent: Обученный DRL агент
            test_data: Тестовые данные
            env_config: Конфигурация среды
            
        Returns:
            Результаты бэктестинга
        """
        self.logger.info("Начало бэктестинга...")
        
        from ..environment.trading_env import create_trading_environment, TradingConfig
        
        # Создание тестовой среды
        trading_config = TradingConfig(
            initial_balance=self.config.initial_capital,
            transaction_fee=self.config.commission,
            slippage=self.config.slippage,
            **(env_config or {})
        )
        
        test_env = create_trading_environment(test_data, trading_config)
        
        # Запуск тестирования
        obs, _ = test_env.reset()
        done = False
        step = 0
        
        portfolio_values = [self.config.initial_capital]
        actions_taken = []
        trades = []
        
        while not done:
            # Получение действия от агента
            action, _ = agent.predict(obs, deterministic=True)
            
            # Выполнение действия
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            # Сохранение данных
            portfolio_values.append(info['portfolio_value'])
            actions_taken.append({
                'step': step,
                'action': action,
                'price': info['current_price'],
                'portfolio_value': info['portfolio_value'],
                'balance': info['balance'],
                'crypto_held': info['crypto_held']
            })
            
            step += 1
        
        # Получение данных о сделках из среды
        if hasattr(test_env, 'trade_history'):
            trades = test_env.trade_history
        
        # Расчет метрик
        portfolio_series = pd.Series(portfolio_values, index=test_data.index[:len(portfolio_values)])
        returns = portfolio_series.pct_change().dropna()
        
        results = self._calculate_metrics(portfolio_series, returns, trades, test_data)
        
        # Сравнение с бенчмарком
        benchmark_results = self._run_benchmark(test_data)
        results['benchmark'] = benchmark_results
        
        self.results = results
        self.logger.info("Бэктестинг завершен")
        
        return results
    
    def _calculate_metrics(
        self, 
        portfolio_values: pd.Series, 
        returns: pd.Series, 
        trades: List[Dict],
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Расчет всех финансовых метрик."""
        
        metrics = {
            # Основные метрики доходности
            'total_return': FinancialMetrics.total_return(portfolio_values),
            'annualized_return': FinancialMetrics.annualized_return(portfolio_values),
            'volatility': FinancialMetrics.volatility(returns),
            
            # Риск-скорректированные метрики
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'sortino_ratio': FinancialMetrics.sortino_ratio(returns, self.config.risk_free_rate),
            'calmar_ratio': FinancialMetrics.calmar_ratio(portfolio_values),
            
            # Метрики просадки
            'max_drawdown': FinancialMetrics.max_drawdown(portfolio_values)[0],
            'max_drawdown_duration': self._calculate_drawdown_duration(portfolio_values),
            
            # Торговые метрики
            'total_trades': len(trades),
            'win_rate': FinancialMetrics.win_rate(trades),
            'profit_factor': FinancialMetrics.profit_factor(trades),
            
            # Дополнительные метрики
            'final_portfolio_value': portfolio_values.iloc[-1],
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            
            # Данные для анализа
            'portfolio_values': portfolio_values,
            'returns': returns,
            'trades': trades
        }
        
        return metrics
    
    def _calculate_drawdown_duration(self, portfolio_values: pd.Series) -> int:
        """Расчет продолжительности максимальной просадки."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        
        # Находим периоды просадки
        is_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        
        for i, in_drawdown in enumerate(is_drawdown):
            if in_drawdown and start is None:
                start = i
            elif not in_drawdown and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # Если просадка продолжается до конца
        if start is not None:
            drawdown_periods.append(len(is_drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _run_benchmark(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Запуск бенчмарка для сравнения."""
        if self.config.benchmark == 'buy_and_hold':
            return self._buy_and_hold_benchmark(test_data)
        elif self.config.benchmark == 'random':
            return self._random_benchmark(test_data)
        elif self.config.benchmark == 'sma_crossover':
            return self._sma_crossover_benchmark(test_data)
        else:
            return {}
    
    def _buy_and_hold_benchmark(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Бенчмарк Buy and Hold."""
        initial_price = test_data['close'].iloc[0]
        final_price = test_data['close'].iloc[-1]
        
        # Покупаем криптовалюту на весь капитал
        crypto_amount = self.config.initial_capital / initial_price
        final_value = crypto_amount * final_price
        
        portfolio_values = (test_data['close'] / initial_price) * self.config.initial_capital
        returns = portfolio_values.pct_change().dropna()
        
        return {
            'strategy': 'Buy and Hold',
            'total_return': (final_value - self.config.initial_capital) / self.config.initial_capital,
            'final_value': final_value,
            'volatility': FinancialMetrics.volatility(returns),
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'max_drawdown': FinancialMetrics.max_drawdown(portfolio_values)[0]
        }
    
    def _random_benchmark(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Случайная торговая стратегия."""
        np.random.seed(42)  # Для воспроизводимости
        
        balance = self.config.initial_capital
        crypto_held = 0.0
        portfolio_values = []
        
        for i, row in test_data.iterrows():
            price = row['close']
            
            # Случайное действие каждые 10 шагов
            if i % 10 == 0:
                action = np.random.choice([0, 1, 2])  # Hold, Buy, Sell
                
                if action == 1 and balance > 100:  # Buy
                    buy_amount = balance * 0.5
                    crypto_bought = buy_amount / price
                    balance -= buy_amount
                    crypto_held += crypto_bought
                elif action == 2 and crypto_held > 0:  # Sell
                    sell_amount = crypto_held * 0.5
                    balance += sell_amount * price
                    crypto_held -= sell_amount
            
            portfolio_value = balance + crypto_held * price
            portfolio_values.append(portfolio_value)
        
        portfolio_series = pd.Series(portfolio_values, index=test_data.index)
        returns = portfolio_series.pct_change().dropna()
        
        return {
            'strategy': 'Random Trading',
            'total_return': (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital,
            'final_value': portfolio_values[-1],
            'volatility': FinancialMetrics.volatility(returns),
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'max_drawdown': FinancialMetrics.max_drawdown(portfolio_series)[0]
        }
    
    def _sma_crossover_benchmark(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Бенчмарк на основе пересечения скользящих средних."""
        data = test_data.copy()
        data['sma_short'] = data['close'].rolling(window=10).mean()
        data['sma_long'] = data['close'].rolling(window=30).mean()
        data = data.dropna()
        
        balance = self.config.initial_capital
        crypto_held = 0.0
        portfolio_values = []
        
        for i, row in data.iterrows():
            price = row['close']
            sma_short = row['sma_short']
            sma_long = row['sma_long']
            
            # Сигнал покупки: короткая SMA пересекает длинную снизу вверх
            if i > 0:
                prev_short = data['sma_short'].iloc[data.index.get_loc(i) - 1]
                prev_long = data['sma_long'].iloc[data.index.get_loc(i) - 1]
                
                if prev_short <= prev_long and sma_short > sma_long and balance > 100:
                    # Покупка
                    crypto_bought = balance / price
                    crypto_held += crypto_bought
                    balance = 0
                elif prev_short >= prev_long and sma_short < sma_long and crypto_held > 0:
                    # Продажа
                    balance = crypto_held * price
                    crypto_held = 0
            
            portfolio_value = balance + crypto_held * price
            portfolio_values.append(portfolio_value)
        
        portfolio_series = pd.Series(portfolio_values, index=data.index)
        returns = portfolio_series.pct_change().dropna()
        
        return {
            'strategy': 'SMA Crossover',
            'total_return': (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital,
            'final_value': portfolio_values[-1],
            'volatility': FinancialMetrics.volatility(returns),
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns, self.config.risk_free_rate),
            'max_drawdown': FinancialMetrics.max_drawdown(portfolio_series)[0]
        }
    
    def print_results(self):
        """Вывод результатов бэктестинга."""
        if not self.results:
            print("Результаты бэктестинга недоступны. Сначала запустите бэктест.")
            return
        
        print("=" * 60)
        print("РЕЗУЛЬТАТЫ БЭКТЕСТИНГА DRL АГЕНТА")
        print("=" * 60)
        
        # Основные метрики
        print(f"Общая доходность: {self.results['total_return']:.2%}")
        print(f"Годовая доходность: {self.results['annualized_return']:.2%}")
        print(f"Волатильность: {self.results['volatility']:.2%}")
        print(f"Коэффициент Шарпа: {self.results['sharpe_ratio']:.2f}")
        print(f"Коэффициент Сортино: {self.results['sortino_ratio']:.2f}")
        print(f"Коэффициент Калмара: {self.results['calmar_ratio']:.2f}")
        
        print(f"\nМаксимальная просадка: {self.results['max_drawdown']:.2%}")
        print(f"Продолжительность макс. просадки: {self.results['max_drawdown_duration']} дней")
        
        print(f"\nВсего сделок: {self.results['total_trades']}")
        print(f"Процент прибыльных: {self.results['win_rate']:.2%}")
        print(f"Фактор прибыли: {self.results['profit_factor']:.2f}")
        
        print(f"\nНачальный капитал: ${self.config.initial_capital:,.2f}")
        print(f"Конечная стоимость: ${self.results['final_portfolio_value']:,.2f}")
        
        # Сравнение с бенчмарком
        if 'benchmark' in self.results:
            bench = self.results['benchmark']
            print(f"\n--- СРАВНЕНИЕ С БЕНЧМАРКОМ ({bench['strategy']}) ---")
            print(f"DRL Agent доходность: {self.results['total_return']:.2%}")
            print(f"{bench['strategy']} доходность: {bench['total_return']:.2%}")
            print(f"Превышение: {(self.results['total_return'] - bench['total_return']):.2%}")
        
        print("=" * 60)


def main():
    """Пример использования бэктестера."""
    import sys
    import os
    
    # Добавляем путь к модулям
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from agents.ppo_agent import create_ppo_agent
    from environment.trading_env import TradingConfig
    import pandas as pd
    import numpy as np
    
    # Создание тестовых данных
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Обеспечение логичности OHLC данных
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    print("Создание и обучение агента...")
    
    # Создание среды для обучения
    from environment.trading_env import create_trading_environment
    config = TradingConfig(initial_balance=10000.0, lookback_window=20)
    train_env = create_trading_environment(test_data[:800], config)  # Первые 800 записей для обучения
    
    # Создание и обучение агента
    agent = create_ppo_agent(train_env)
    agent.create_model()
    agent.train(total_timesteps=5000)  # Короткое обучение для примера
    
    print("Запуск бэктестинга...")
    
    # Бэктестинг на последних 200 записях
    backtester = Backtester()
    results = backtester.run_backtest(agent, test_data[800:])
    
    # Вывод результатов
    backtester.print_results()


if __name__ == "__main__":
    main()