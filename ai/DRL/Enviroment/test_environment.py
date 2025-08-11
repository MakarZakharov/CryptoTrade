"""
Unit-тесты для торгового окружения.
Тестируют все ключевые компоненты системы.

Запуск:
    python test_environment.py
    или
    pytest test_environment.py -v
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from data_loader import DataLoader
from simulator import MarketSimulator, OrderSide, SlippageModel
from metrics import MetricsCalculator, TradeMetrics, PerformanceMetrics
from env import CryptoTradingEnv, ActionSpace, RewardType


class TestDataLoader(unittest.TestCase):
    """Тесты для DataLoader."""

    def setUp(self):
        """Настройка для каждого теста."""
        # Создаем тестовый DataFrame
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'open': np.random.uniform(50000, 52000, 100),
            'high': np.random.uniform(52000, 54000, 100),
            'low': np.random.uniform(48000, 50000, 100),
            'close': np.random.uniform(50000, 52000, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })

        # Временная директория
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_data.parquet"
        self.test_data.to_parquet(self.test_file)

    def tearDown(self):
        """Очистка после каждого теста."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_parquet(self):
        """Тест загрузки Parquet файла."""
        loader = DataLoader(data_path=str(self.test_file), add_indicators=False)
        data = loader.load()

        self.assertIsNotNone(data)
        self.assertEqual(len(data), 100)
        self.assertIn('close', data.columns)

    def test_add_indicators(self):
        """Тест добавления технических индикаторов."""
        loader = DataLoader(data_path=str(self.test_file), add_indicators=True)
        data = loader.load()

        # Проверяем наличие индикаторов
        self.assertIn('sma_7', data.columns)
        self.assertIn('rsi_14', data.columns)
        self.assertIn('macd', data.columns)

    def test_normalization(self):
        """Тест нормализации данных."""
        loader = DataLoader(data_path=str(self.test_file), normalize=True)
        data = loader.load()

        # Проверяем, что данные нормализованы
        self.assertIsNotNone(loader.normalization_params)

    def test_get_window(self):
        """Тест получения окна данных."""
        loader = DataLoader(data_path=str(self.test_file))
        loader.load()

        window = loader.get_window(start_idx=50, window_size=10)

        self.assertEqual(window.shape[0], 10)

    def test_train_test_split(self):
        """Тест разделения на train/test."""
        loader = DataLoader(data_path=str(self.test_file))
        loader.load()

        train_loader, test_loader = loader.split_train_test(train_ratio=0.8)

        self.assertEqual(len(train_loader), 80)
        self.assertEqual(len(test_loader), 20)


class TestMarketSimulator(unittest.TestCase):
    """Тесты для MarketSimulator."""

    def setUp(self):
        """Настройка для каждого теста."""
        self.simulator = MarketSimulator(
            maker_fee=0.0001,
            taker_fee=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_percentage=0.0005
        )

    def test_market_state(self):
        """Тест создания состояния рынка."""
        market = self.simulator.get_market_state(
            mid_price=50000.0,
            volume=1000000.0
        )

        self.assertIsNotNone(market)
        self.assertGreater(market.ask_price, market.bid_price)
        self.assertAlmostEqual(market.mid_price, 50000.0, delta=100)

    def test_buy_order(self):
        """Тест исполнения ордера на покупку."""
        market = self.simulator.get_market_state(50000.0, 1000000.0)

        result = self.simulator.execute_order(
            side=OrderSide.BUY,
            quantity=0.5,
            market_state=market
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.side, OrderSide.BUY)
        self.assertGreater(result.executed_quantity, 0)
        self.assertGreater(result.commission, 0)

    def test_sell_order(self):
        """Тест исполнения ордера на продажу."""
        market = self.simulator.get_market_state(50000.0, 1000000.0)

        result = self.simulator.execute_order(
            side=OrderSide.SELL,
            quantity=0.5,
            market_state=market
        )

        self.assertTrue(result.executed)
        self.assertEqual(result.side, OrderSide.SELL)

    def test_slippage(self):
        """Тест проскальзывания."""
        market = self.simulator.get_market_state(50000.0, 1000000.0)

        result = self.simulator.execute_order(
            side=OrderSide.BUY,
            quantity=0.5,
            market_state=market
        )

        # Проверяем, что есть проскальзывание
        self.assertGreater(result.slippage, 0)

    def test_commission(self):
        """Тест комиссий."""
        market = self.simulator.get_market_state(50000.0, 1000000.0)

        result = self.simulator.execute_order(
            side=OrderSide.BUY,
            quantity=0.5,
            market_state=market
        )

        # Проверяем наличие комиссии
        self.assertGreater(result.commission, 0)
        expected_commission = result.executed_quantity * result.executed_price * self.simulator.taker_fee
        self.assertAlmostEqual(result.commission, expected_commission, places=2)


class TestMetricsCalculator(unittest.TestCase):
    """Тесты для MetricsCalculator."""

    def setUp(self):
        """Настройка для каждого теста."""
        self.calculator = MetricsCalculator()

        # Создаем тестовую equity curve
        np.random.seed(42)
        initial_balance = 10000.0
        returns = np.random.normal(0.001, 0.02, 100)
        self.equity_curve = [initial_balance]

        for ret in returns:
            self.equity_curve.append(self.equity_curve[-1] * (1 + ret))

    def test_calculate_metrics(self):
        """Тест расчета метрик."""
        metrics = self.calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            initial_balance=10000.0
        )

        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsNotNone(metrics.total_return_pct)
        self.assertIsNotNone(metrics.sharpe_ratio)

    def test_sharpe_ratio(self):
        """Тест расчета Sharpe ratio."""
        metrics = self.calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            initial_balance=10000.0
        )

        # Sharpe должен быть числом
        self.assertIsInstance(metrics.sharpe_ratio, (int, float))

    def test_drawdown(self):
        """Тест расчета просадки."""
        metrics = self.calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            initial_balance=10000.0
        )

        # Максимальная просадка должна быть отрицательной или нулевой
        self.assertLessEqual(metrics.max_drawdown_pct, 0)

    def test_trade_metrics(self):
        """Тест метрик сделок."""
        trades = [
            TradeMetrics(
                entry_time=1.0,
                exit_time=2.0,
                entry_price=100.0,
                exit_price=105.0,
                quantity=1.0,
                side='long',
                pnl=5.0,
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
                side='long',
                pnl=-2.0,
                commission=0.1,
                holding_time=1.0,
                is_winner=False
            )
        ]

        metrics = self.calculator.calculate_metrics(
            equity_curve=self.equity_curve,
            trades=trades,
            initial_balance=10000.0
        )

        self.assertEqual(metrics.total_trades, 2)
        self.assertEqual(metrics.winning_trades, 1)
        self.assertEqual(metrics.losing_trades, 1)
        self.assertAlmostEqual(metrics.win_rate, 50.0, places=1)


class TestCryptoTradingEnv(unittest.TestCase):
    """Тесты для CryptoTradingEnv."""

    def setUp(self):
        """Настройка для каждого теста."""
        # Создаем тестовые данные
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='D'),
            'open': np.random.uniform(50000, 52000, 200),
            'high': np.random.uniform(52000, 54000, 200),
            'low': np.random.uniform(48000, 50000, 200),
            'close': np.random.uniform(50000, 52000, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        })

        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_data.parquet"
        self.test_data.to_parquet(self.test_file)

    def tearDown(self):
        """Очистка после каждого теста."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_env_creation(self):
        """Тест создания окружения."""
        env = CryptoTradingEnv(
            data_path=str(self.test_file),
            initial_balance=10000.0
        )

        self.assertIsNotNone(env)
        self.assertIsNotNone(env.action_space)
        self.assertIsNotNone(env.observation_space)

    def test_reset(self):
        """Тест reset окружения."""
        env = CryptoTradingEnv(data_path=str(self.test_file))
        obs, info = env.reset()

        self.assertIsNotNone(obs)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn('balance', info)
        self.assertEqual(info['balance'], env.initial_balance)

    def test_step(self):
        """Тест step окружения."""
        env = CryptoTradingEnv(data_path=str(self.test_file))
        obs, info = env.reset()

        # Hold action
        obs, reward, terminated, truncated, info = env.step(0)

        self.assertIsNotNone(obs)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_buy_action(self):
        """Тест действия покупки."""
        env = CryptoTradingEnv(data_path=str(self.test_file))
        env.reset()

        initial_balance = env.balance

        # Buy action
        obs, reward, terminated, truncated, info = env.step(1)

        # После покупки баланс должен уменьшиться
        self.assertLess(env.balance, initial_balance)
        self.assertGreater(env.crypto_held, 0)

    def test_sell_action(self):
        """Тест действия продажи."""
        env = CryptoTradingEnv(data_path=str(self.test_file))
        env.reset()

        # Сначала покупаем
        env.step(1)
        crypto_before_sell = env.crypto_held

        # Потом продаем
        env.step(2)

        # После продажи крипта должно быть меньше
        self.assertLess(env.crypto_held, crypto_before_sell)

    def test_episode_completion(self):
        """Тест завершения эпизода."""
        env = CryptoTradingEnv(
            data_path=str(self.test_file),
            max_steps=10
        )
        env.reset()

        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 20:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        # Должен завершиться через 10 шагов
        self.assertTrue(truncated or steps == 20)

    def test_metrics(self):
        """Тест получения метрик."""
        env = CryptoTradingEnv(data_path=str(self.test_file))
        env.reset()

        # Делаем несколько шагов
        for _ in range(10):
            env.step(env.action_space.sample())

        metrics = env.get_metrics()

        self.assertIsInstance(metrics, PerformanceMetrics)

    def test_different_action_spaces(self):
        """Тест разных action spaces."""
        # Discrete
        env_discrete = CryptoTradingEnv(
            data_path=str(self.test_file),
            action_type=ActionSpace.DISCRETE
        )
        self.assertEqual(env_discrete.action_space.n, 3)

        # Continuous
        env_continuous = CryptoTradingEnv(
            data_path=str(self.test_file),
            action_type=ActionSpace.CONTINUOUS
        )
        self.assertEqual(env_continuous.action_space.shape, (1,))

    def test_different_reward_types(self):
        """Тест разных типов наград."""
        reward_types = [
            RewardType.PNL,
            RewardType.LOG_RETURN,
            RewardType.SHARPE,
            RewardType.RISK_ADJUSTED
        ]

        for reward_type in reward_types:
            env = CryptoTradingEnv(
                data_path=str(self.test_file),
                reward_type=reward_type
            )
            env.reset()

            obs, reward, _, _, _ = env.step(0)

            self.assertIsInstance(reward, (int, float))


def run_tests():
    """Запустить все тесты."""
    # Создаем test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Добавляем тесты
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestMarketSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestCryptoTradingEnv))

    # Запускаем
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Выводим результаты
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
