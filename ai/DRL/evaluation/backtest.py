"""Модуль бектестинга для DRL агентов."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from ..agents.base_agent import BaseAgent
from ..environments import TradingEnv
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger, TradingMetrics


class DRLBacktester:
    """
    Система бектестинга для DRL агентов.
    
    Предоставляет комплексное тестирование обученных агентов
    на исторических данных с детальной аналитикой.
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация бектестера.
        
        Args:
            agent: обученный DRL агент
            config: торговая конфигурация
            logger: логгер
        """
        self.agent = agent
        self.config = config
        self.logger = logger or DRLLogger("drl_backtester")
        
        # Результаты бектеста
        self.results: Dict[str, Any] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        
        # Метрики
        self.trading_metrics = TradingMetrics()
        
        self.logger.info("DRL Backtester инициализирован")
    
    def run_backtest(
        self,
        test_data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        deterministic: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Запуск полного бектеста.
        
        Args:
            test_data: тестовые данные (опционально)
            start_date: начальная дата
            end_date: конечная дата
            deterministic: использовать детерминистичную политику
            save_results: сохранять результаты
            
        Returns:
            Результаты бектеста
        """
        self.logger.info("Запуск бектеста DRL агента...")
        start_time = datetime.now()
        
        try:
            # Подготовка среды
            test_env = self._prepare_test_environment(test_data, start_date, end_date)
            
            # Выполнение бектеста
            episode_results = self._run_backtest_episode(test_env, deterministic)
            
            # Анализ результатов
            analysis_results = self._analyze_results(episode_results)
            
            # Составление финального отчета
            self.results = self._compile_final_results(
                episode_results, 
                analysis_results,
                start_time
            )
            
            if save_results:
                self._save_results()
            
            self.logger.info("Бектест завершен успешно")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении бектеста: {e}")
            raise
    
    def _prepare_test_environment(
        self,
        test_data: Optional[pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> TradingEnv:
        """Подготовка тестовой среды."""
        self.logger.info("Подготовка тестовой среды...")
        
        # Создаем тестовую среду
        test_env = TradingEnv(self.config, data=test_data, logger=self.logger)
        
        # Устанавливаем тестовые данные
        test_env.set_data_split("test")
        
        # Фильтрация по датам если нужно
        if start_date or end_date:
            test_env = self._filter_data_by_dates(test_env, start_date, end_date)
        
        data_info = test_env.get_data_info()
        self.logger.info(f"Тестовые данные: {data_info['test_samples']} образцов")
        self.logger.info(f"Период: {start_date or 'начало'} - {end_date or 'конец'}")
        
        return test_env
    
    def _filter_data_by_dates(
        self, 
        env: TradingEnv, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> TradingEnv:
        """Фильтрация данных по датам."""
        # Эта функциональность может быть расширена
        # для поддержки фильтрации данных по конкретным датам
        return env
    
    def _run_backtest_episode(
        self, 
        test_env: TradingEnv, 
        deterministic: bool
    ) -> Dict[str, Any]:
        """Выполнение одного эпизода бектеста."""
        self.logger.info("Выполнение бектест эпизода...")
        
        # Сброс среды
        obs, info = test_env.reset()
        
        # Инициализация отслеживания
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'info': [],
            'portfolio_values': [],
            'prices': []
        }
        
        step = 0
        total_reward = 0
        done = False
        
        while not done:
            # Предсказание действия
            action = self.agent.predict(obs, deterministic=deterministic)
            
            # Выполнение шага
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            # Сохранение данных
            episode_data['observations'].append(obs.copy())
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['info'].append(info.copy())
            episode_data['portfolio_values'].append(
                info.get('portfolio', {}).get('total_value', 0)
            )
            episode_data['prices'].append(info.get('price', 0))
            
            total_reward += reward
            step += 1
            
            if step % 1000 == 0:
                self.logger.debug(f"Шаг {step}, награда: {reward:.4f}")
        
        # Финальные метрики эпизода
        final_info = test_env.get_episode_summary()
        
        episode_results = {
            'episode_data': episode_data,
            'final_info': final_info,
            'total_steps': step,
            'total_reward': total_reward,
            'final_portfolio_value': episode_data['portfolio_values'][-1] if episode_data['portfolio_values'] else 0
        }
        
        self.logger.info(f"Эпизод завершен: {step} шагов, итоговая награда: {total_reward:.4f}")
        
        return episode_results
    
    def _analyze_results(self, episode_results: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ результатов бектеста."""
        self.logger.info("Анализ результатов бектеста...")
        
        episode_data = episode_results['episode_data']
        
        # Базовые метрики
        portfolio_values = np.array(episode_data['portfolio_values'])
        prices = np.array(episode_data['prices'])
        rewards = np.array(episode_data['rewards'])
        
        # Доходность
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]  # Удаляем NaN
        
        # Базовые статистики
        initial_value = portfolio_values[0] if len(portfolio_values) > 0 else self.config.initial_balance
        final_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_value
        total_return = (final_value - initial_value) / initial_value
        
        # Buy & Hold сравнение
        initial_price = prices[0] if len(prices) > 0 else 1
        final_price = prices[-1] if len(prices) > 0 else initial_price
        buy_hold_return = (final_price - initial_price) / initial_price
        
        # Риск-метрики
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = self.trading_metrics.sharpe_ratio(returns) if len(returns) > 1 else 0
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # Торговые метрики
        trade_analysis = self._analyze_trades(episode_data['info'])
        
        analysis = {
            'performance_metrics': {
                'total_return': float(total_return),
                'total_return_pct': float(total_return * 100),
                'annualized_return': float(total_return * 252 / len(portfolio_values)) if len(portfolio_values) > 0 else 0,
                'buy_hold_return': float(buy_hold_return),
                'buy_hold_return_pct': float(buy_hold_return * 100),
                'alpha': float(total_return - buy_hold_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'max_drawdown_pct': float(max_drawdown * 100),
                'final_portfolio_value': float(final_value),
                'initial_portfolio_value': float(initial_value)
            },
            'trading_metrics': trade_analysis,
            'reward_metrics': {
                'total_reward': float(np.sum(rewards)),
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards))
            }
        }
        
        return analysis
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Расчет максимальной просадки."""
        if len(portfolio_values) == 0:
            return 0.0
        
        # Кумулятивный максимум
        cummax = np.maximum.accumulate(portfolio_values)
        
        # Просадка
        drawdown = (cummax - portfolio_values) / cummax
        
        return float(np.max(drawdown))
    
    def _analyze_trades(self, info_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ торговых сделок."""
        trades = []
        positions = []
        
        for info in info_history:
            if 'trade' in info:
                trades.append(info['trade'])
            
            portfolio = info.get('portfolio', {})
            if portfolio:
                positions.append({
                    'step': info.get('step', 0),
                    'position_size': portfolio.get('position_size', 0),
                    'total_value': portfolio.get('total_value', 0)
                })
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_return': 0.0,
                'avg_winning_trade': 0.0,
                'avg_losing_trade': 0.0
            }
        
        # Анализ сделок
        trade_returns = []
        winning_trades = []
        losing_trades = []
        
        for trade in trades:
            if isinstance(trade, dict) and 'pnl' in trade:
                pnl = trade['pnl']
                if pnl != 0:
                    trade_returns.append(pnl)
                    if pnl > 0:
                        winning_trades.append(pnl)
                    else:
                        losing_trades.append(pnl)
        
        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
        
        return {
            'total_trades': len(trades),
            'profitable_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_trade_return': float(np.mean(trade_returns)) if trade_returns else 0,
            'avg_winning_trade': float(np.mean(winning_trades)) if winning_trades else 0,
            'avg_losing_trade': float(np.mean(losing_trades)) if losing_trades else 0,
            'largest_win': float(np.max(winning_trades)) if winning_trades else 0,
            'largest_loss': float(np.min(losing_trades)) if losing_trades else 0
        }
    
    def _compile_final_results(
        self, 
        episode_results: Dict[str, Any],
        analysis_results: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """Составление финального отчета."""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        final_results = {
            'metadata': {
                'agent_type': self.agent.__class__.__name__,
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'initial_balance': self.config.initial_balance,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'execution_time_seconds': execution_time,
                'total_steps': episode_results['total_steps']
            },
            'performance': analysis_results['performance_metrics'],
            'trading': analysis_results['trading_metrics'],
            'rewards': analysis_results['reward_metrics'],
            'summary': {
                'success': True,
                'total_return_pct': analysis_results['performance_metrics']['total_return_pct'],
                'vs_buy_hold': analysis_results['performance_metrics']['alpha'] * 100,
                'sharpe_ratio': analysis_results['performance_metrics']['sharpe_ratio'],
                'max_drawdown_pct': analysis_results['performance_metrics']['max_drawdown_pct'],
                'win_rate': analysis_results['trading_metrics']['win_rate'] * 100,
                'total_trades': analysis_results['trading_metrics']['total_trades']
            }
        }
        
        return final_results
    
    def _save_results(self, filename: Optional[str] = None):
        """Сохранение результатов бектеста."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{self.config.symbol}_{timestamp}.json"
        
        # Создаем директорию если не существует
        results_dir = Path("CryptoTrade/ai/DRL/evaluation/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Результаты бектеста сохранены: {filepath}")
    
    def print_results(self):
        """Вывод результатов бектеста в консоль."""
        if not self.results:
            self.logger.warning("Нет результатов для отображения")
            return
        
        print("\n" + "="*70)
        print("РЕЗУЛЬТАТЫ БЕКТЕСТА DRL АГЕНТА")
        print("="*70)
        
        # Метаданные
        meta = self.results['metadata']
        print(f"Агент: {meta['agent_type']}")
        print(f"Символ: {meta['symbol']} ({meta['timeframe']})")
        print(f"Начальный баланс: ${meta['initial_balance']:,.2f}")
        print(f"Время выполнения: {meta['execution_time_seconds']:.2f} сек")
        print(f"Всего шагов: {meta['total_steps']:,}")
        
        # Производительность
        print(f"\n📊 ПРОИЗВОДИТЕЛЬНОСТЬ:")
        perf = self.results['performance']
        print(f"   Итоговая доходность: {perf['total_return_pct']:.2f}%")
        print(f"   Buy & Hold доходность: {perf['buy_hold_return_pct']:.2f}%")
        print(f"   Альфа (превышение): {perf['alpha']*100:.2f}%")
        print(f"   Коэффициент Шарпа: {perf['sharpe_ratio']:.2f}")
        print(f"   Максимальная просадка: {perf['max_drawdown_pct']:.2f}%")
        print(f"   Волатильность: {perf['volatility']:.2f}")
        
        # Торговля
        print(f"\n💼 ТОРГОВАЯ АКТИВНОСТЬ:")
        trading = self.results['trading']
        print(f"   Всего сделок: {trading['total_trades']}")
        print(f"   Прибыльные сделки: {trading['profitable_trades']}")
        print(f"   Убыточные сделки: {trading['losing_trades']}")
        print(f"   Винрейт: {trading['win_rate']*100:.1f}%")
        print(f"   Profit Factor: {trading['profit_factor']:.2f}")
        print(f"   Средняя прибыль сделки: ${trading['avg_winning_trade']:.2f}")
        print(f"   Средний убыток сделки: ${trading['avg_losing_trade']:.2f}")
        
        # Награды
        print(f"\n🎯 МЕТРИКИ НАГРАД:")
        rewards = self.results['rewards']
        print(f"   Общая награда: {rewards['total_reward']:.4f}")
        print(f"   Средняя награда: {rewards['mean_reward']:.4f}")
        print(f"   Лучшая награда: {rewards['max_reward']:.4f}")
        print(f"   Худшая награда: {rewards['min_reward']:.4f}")
        
        print("="*70)
        
        # Краткая сводка
        summary = self.results['summary']
        print(f"\n🏆 ИТОГО: {summary['total_return_pct']:.2f}% доходности за {meta['total_steps']} шагов")
        print(f"    Превышение над Buy&Hold: {summary['vs_buy_hold']:.2f}%")
        print(f"    Риск-скорректированная доходность (Sharpe): {summary['sharpe_ratio']:.2f}")
    
    def get_results(self) -> Dict[str, Any]:
        """Получение результатов бектеста."""
        return self.results.copy()
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Сравнение с базовой стратегией.
        
        Args:
            baseline_results: результаты базовой стратегии
            
        Returns:
            Сравнительный анализ
        """
        if not self.results:
            raise ValueError("Сначала выполните бектест")
        
        comparison = {
            'drl_return': self.results['performance']['total_return_pct'],
            'baseline_return': baseline_results.get('total_return_pct', 0),
            'outperformance': self.results['performance']['total_return_pct'] - baseline_results.get('total_return_pct', 0),
            'drl_sharpe': self.results['performance']['sharpe_ratio'],
            'baseline_sharpe': baseline_results.get('sharpe_ratio', 0),
            'drl_max_dd': self.results['performance']['max_drawdown_pct'],
            'baseline_max_dd': baseline_results.get('max_drawdown_pct', 0)
        }
        
        return comparison


def run_quick_backtest(
    agent: BaseAgent,
    config: TradingConfig,
    test_data: Optional[pd.DataFrame] = None,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Быстрый запуск бектеста.
    
    Args:
        agent: обученный агент
        config: торговая конфигурация
        test_data: тестовые данные
        deterministic: детерминистичная политика
        
    Returns:
        Результаты бектеста
    """
    backtester = DRLBacktester(agent, config)
    return backtester.run_backtest(
        test_data=test_data,
        deterministic=deterministic,
        save_results=False
    )